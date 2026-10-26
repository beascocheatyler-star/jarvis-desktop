"""In-process replacement for the FastAPI layer.

The web build called ``/api/chat`` over localhost. Here the GUI calls
``VaultController.ask()`` directly: same engine, same gate, same citations, no
HTTP, no port, no server process to start or leak. That removes the entire
class of "the app is open but the backend died" failure.

Threading model: the GUI thread must never block. ``open_vault`` and ``ask``
are therefore designed to be run on a worker thread, and they report progress
through callbacks rather than returning only at the end.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.config import Settings
from app.core.embedding import get_embedder
from app.core.engine import Answer, ChatEngine, LLMUnavailableError
from app.core.metadata import InvalidFilterError, ScopeFilter
from app.core.persona import Persona
from app.core.store import DocumentStore

from desktop.appsettings import AppSettings, load_api_key
from desktop.vault import IndexReport, VaultIndexer

logger = logging.getLogger(__name__)

# How many prior turns to send. The engine caps this too; keeping our own copy
# bounded means a long session cannot grow the per-message cost without limit.
MAX_HISTORY_TURNS = 8


def vault_persona(vault_name: str) -> Persona:
    """The assistant's identity when pointed at a personal vault.

    The bundled presets describe a course TA and a support agent. Neither fits
    a personal knowledge base, and the ``topic`` string feeds the refusal
    message, so a wrong one produces confusing rejections.
    """
    return Persona(
        name="Jarvis",
        topic="the notes, documents, and files in the {} vault".format(vault_name),
        style=(
            "Be direct, concrete, and brief. Lead with the answer, then the "
            "supporting detail. Use short paragraphs and bulleted lists rather "
            "than long prose blocks. When the notes conflict or are incomplete, "
            "say so plainly instead of smoothing it over. Never pad with "
            "restatements of the question or summaries of what you are about to say."
        ),
        greeting=(
            "Vault indexed. Ask me anything in it — I answer only from your "
            "notes and cite which ones."
        ),
        suggested_questions=[],
    )


# Words that describe *what to do with* an answer rather than what it is
# about. The gate scores a question as sqrt(bm25_normalised x coverage), where
# coverage is the fraction of content words found anywhere in the corpus. Words
# like "draft", "email", "biggest" never appear in a note, so a naturally
# phrased request drags its own coverage down and gets refused — while the same
# intent in two keywords passes easily.
#
# Measured on a real vault: "academic clearance risk" scored 0.56 and passed,
# while "Based on my academic clearance plan and triage notes, what is the
# single biggest risk to my semester?" scored 0.33 and was refused — despite
# retrieving the correct note at full coverage. Four of five naturally phrased
# questions were refused this way.
#
# These are stripped from the *gate query only*. The model still receives the
# user's full question, so "draft an email" still produces an email.
_INSTRUCTION_WORDS = frozenset(
    """
    according across actual actually add additional advice advise all also always
    analyse analyze answer anything appreciate ask asking based basically best
    better biggest brief briefly bullet bullets call can check clear compare
    compile concise could couple create critical currently detail detailed
    details discuss draft due each essay evaluate exactly explain explanation
    first focus following format full generate give giving good greatest help
    high highest how identify important include including inform information
    into itemise itemize key kind layout least less level list listing little
    long look looking lot main major make making many maybe mean means mention
    might more most much need needs new next note notes outline overall overview
    paragraph part particular perhaps plan please point points possible prepare
    pretty provide quick quickly rank rate really recap recommend relevant
    remind reply report request require requires review rewrite right rundown
    say says search see short should show significant simple simply single
    smallest some something soon soonest sort stuff suggest summarise summarize
    summary table tell thing things think thoughts tldr top true try update
    urgent use using usually want way ways whole worst would write writing
    """.split()
)


def gate_query(question: str) -> str:
    """Strip instruction words so the gate scores topic, not phrasing.

    Returns the original question when stripping would leave nothing — a
    question made entirely of instruction words has no topic to score, and
    an empty query would score zero and refuse everything.
    """
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", question)
    kept = [w for w in words if w.lower() not in _INSTRUCTION_WORDS]
    if not kept:
        return question
    return " ".join(kept)


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    text: str
    answer: Optional[Answer] = None


@dataclass
class VaultState:
    """Everything the UI needs to render the current vault at a glance."""

    path: str = ""
    name: str = ""
    document_count: int = 0
    chunk_count: int = 0
    sectors: List[str] = field(default_factory=list)
    tree: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    last_report: Optional[IndexReport] = None


class VaultController:
    """Owns the store, indexer, and engine for one open vault."""

    def __init__(self, app_settings: AppSettings):
        self.app_settings = app_settings
        self.store: Optional[DocumentStore] = None
        self.indexer: Optional[VaultIndexer] = None
        self.engine: Optional[ChatEngine] = None
        # Built lazily by `agent()`; reset whenever a vault is opened, since
        # the tools close over the controller's vault state.
        self._agent = None
        self.settings: Optional[Settings] = None
        self.persona: Persona = vault_persona("your")
        self.history: List[ChatTurn] = []
        self.state = VaultState()
        self._lock = threading.RLock()

    # --- lifecycle ------------------------------------------------------

    def _build_settings(self) -> Settings:
        """Core Settings driven by the GUI preferences, not by a .env file."""
        s = AppSettings.load() if self.app_settings is None else self.app_settings
        settings = Settings(
            anthropic_api_key=load_api_key(),
            model=s.model,
            max_tokens=s.max_tokens,
            effort=s.effort,
            top_k=s.top_k,
            relevance_floor=s.relevance_floor,
            dense_rescue_floor=s.dense_rescue_floor,
            embedding_backend=s.embedding_backend,
            # Must come from the app settings too: the core default is 1024,
            # but the sentence-transformers model is 384-dimensional and a
            # mismatch silently produces a different vector space.
            embedding_dimensions=s.embedding_dimensions,
            data_dir=s.data_dir_for_vault(),
        )
        settings.ensure_dirs()
        return settings

    def open_vault(
        self,
        path: str,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> VaultState:
        """Point the app at a folder and bring its index up to date."""
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError("Not a folder: {}".format(root))

        with self._lock:
            self.app_settings.remember_vault(str(root))
            self.app_settings.save()

            self.settings = self._build_settings()
            embedder = get_embedder(
                self.settings.embedding_backend,
                self.settings.embedding_dimensions,
            )
            self.store = DocumentStore(
                data_dir=self.settings.data_dir,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
                embedder=embedder,
                rrf_k=self.settings.rrf_k,
                lexical_weight=self.settings.lexical_weight,
                dense_weight=self.settings.dense_weight,
            )
            self.indexer = VaultIndexer(
                self.store, root, excluded_folders=self.app_settings.excluded_folders
            )
            self.engine = ChatEngine(self.settings, self.store)
            self._agent = None  # tools bind to the vault that is open
            self.persona = vault_persona(root.name)
            self.history = []

            report = self.indexer.scan(progress=progress)
            self.state = self._snapshot(root, report)
            return self.state

    def rescan(
        self, progress: Optional[Callable[[int, int, str], None]] = None
    ) -> VaultState:
        if not self.indexer or not self.store:
            raise RuntimeError("No vault is open.")
        with self._lock:
            report = self.indexer.scan(progress=progress)
            self.state = self._snapshot(Path(self.state.path), report)
            return self.state

    def reindex_file(self, path: Path) -> bool:
        """Re-index one note (file-watcher path). Returns True if it changed."""
        if not self.indexer:
            return False
        with self._lock:
            changed = self.indexer.reindex_file(path)
            if changed:
                self.state = self._snapshot(Path(self.state.path), self.state.last_report)
            return changed

    def _snapshot(self, root: Path, report: Optional[IndexReport]) -> VaultState:
        assert self.store is not None and self.indexer is not None
        return VaultState(
            path=str(root),
            name=root.name,
            document_count=len(self.store.list_documents()),
            chunk_count=self.store.total_chunks,
            sectors=self.indexer.sectors(),
            tree=self.indexer.tree(),
            last_report=report,
        )

    @property
    def is_open(self) -> bool:
        return self.store is not None and self.engine is not None

    @property
    def has_api_key(self) -> bool:
        return bool(self.settings and self.settings.has_api_key)

    # --- scoping --------------------------------------------------------

    def build_scope(
        self,
        sectors: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> Optional[ScopeFilter]:
        """Turn a sidebar selection into a metadata filter.

        Selecting specific notes is the narrower intent, so it wins over a
        sector selection rather than being unioned with it — widening a scope
        the user thought they had narrowed is the dangerous direction.
        """
        if doc_ids:
            return ScopeFilter.parse(doc_ids=list(doc_ids))
        if sectors:
            try:
                # `raw` is positional in ScopeFilter.parse; a list of values
                # for one key is the OR form.
                return ScopeFilter.parse({"sector": list(sectors)})
            except InvalidFilterError as exc:
                logger.warning("Bad scope filter: %s", exc)
                return None
        return None

    # --- chat -----------------------------------------------------------

    def ask(
        self,
        question: str,
        sectors: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> Answer:
        """Retrieve, gate, and answer. Safe to call from a worker thread."""
        if not self.engine:
            raise RuntimeError("Open a vault first.")

        question = (question or "").strip()
        if not question:
            raise ValueError("Empty question.")

        scope = self.build_scope(sectors=sectors, doc_ids=doc_ids)
        history = self._history_payload()

        self.history.append(ChatTurn(role="user", text=question))

        # The engine both retrieves and generates from the same string, so the
        # instruction-stripped form is passed for scoring and the user's real
        # question is restored in the message history the model reads. Without
        # this, "draft an email about X" is refused for containing the words
        # "draft" and "email"; with it, the gate scores "X".
        retrieval_query = gate_query(question)
        if retrieval_query != question:
            history = history + [
                {
                    "role": "user",
                    "content": (
                        "For context, my full request is: {}\n\n"
                        "Answer it using the retrieved passages.".format(question)
                    ),
                }
            ]

        try:
            answer = self.engine.answer(
                question=retrieval_query,
                persona=self.persona,
                history=history,
                scope=scope,
            )
        except LLMUnavailableError as exc:
            # Surfaced as a normal assistant turn: a missing key is a setup
            # problem, not a crash, and the message explains the fix.
            answer = Answer(
                text=str(exc),
                in_scope=False,
                decision="error",
                reason="LLM unavailable",
            )
        self.history.append(
            ChatTurn(role="assistant", text=answer.text, answer=answer)
        )
        return answer

    def _history_payload(self) -> List[Dict[str, str]]:
        turns = [
            {"role": t.role, "content": t.text}
            for t in self.history
            if t.text.strip()
        ]
        return turns[-(MAX_HISTORY_TURNS * 2) :]

    def clear_history(self) -> None:
        self.history = []
        if self._agent is not None:
            # Drop any half-finished tool conversation, including a pending
            # approval, so a new chat cannot resume the previous one's action.
            self._agent.clear()

    # --- agent ----------------------------------------------------------

    def agent(self):
        """The tool-using agent for this vault, created on first use."""
        if not self.is_open:
            raise RuntimeError("Open a vault first.")
        if self._agent is None:
            from desktop.agent import VaultAgent

            self._agent = VaultAgent(self)
        return self._agent

    def act(self, request: str, on_step=None, auto_approve: bool = False):
        """Run the agent on a request that may read *and* write the vault.

        Separate from ``ask``: that path retrieves, gates, and answers in one
        shot, which is right for a question and wrong for an instruction —
        "create a note about today's workout" matches nothing in the corpus,
        so the gate refuses it. The agent searches on its own instead.
        """
        request = (request or "").strip()
        if not request:
            raise ValueError("Empty request.")

        agent = self.agent()
        history = self._history_payload()
        self.history.append(ChatTurn(role="user", text=request))
        result = agent.run(
            request, history=history, on_step=on_step, auto_approve=auto_approve
        )
        if result.text:
            self.history.append(ChatTurn(role="assistant", text=result.text))
        return result

    def resume_agent(self, approved: bool, on_step=None, auto_approve: bool = False):
        """Continue the agent after the user approves or declines an action."""
        result = self.agent().resume(
            approved, on_step=on_step, auto_approve=auto_approve
        )
        if result.text:
            self.history.append(ChatTurn(role="assistant", text=result.text))
        return result

    # --- browsing and editing ------------------------------------------
    #
    # The index stores chunks, not files. Reading a note therefore goes to
    # disk rather than reconstructing it from chunks: chunks are lowercased
    # for some formats, overlap at the edges, and carry the synthetic title
    # header. Disk is the source of truth; the index is a search structure.

    def _resolve(self, relative: str) -> Path:
        """Vault-relative path -> absolute, refusing anything outside the vault.

        ``relative`` can arrive from stored metadata, so it is not trusted.
        A note named ``../../.ssh/id_rsa`` must not be readable, and a
        symlink pointing outside the vault must not be followed.
        """
        if not self.state.path:
            raise RuntimeError("No vault is open.")
        root = Path(self.state.path).resolve()
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise ValueError("Path is outside the vault: {}".format(relative))
        return target

    def read_note(self, relative: str) -> str:
        """Current text of a note, straight from disk."""
        path = self._resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        if path.suffix.lower() not in {".md", ".txt", ".csv", ".json"}:
            raise ValueError(
                "{} is not a text file — open it in Finder.".format(path.name)
            )
        return path.read_text("utf-8", errors="replace")

    def write_note(self, relative: str, text: str) -> bool:
        """Save a note and re-index it immediately.

        Written via a temporary file and atomic replace so an interrupted
        save cannot leave a half-written note in the vault.
        """
        path = self._resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        changed = self.reindex_file(path)
        self.state = self._snapshot(Path(self.state.path), self.state.last_report)
        return changed

    def create_note(self, relative: str, text: str = "") -> str:
        """Create a new note. Returns the path actually used.

        A numeric suffix is appended rather than overwriting an existing
        note — silently replacing a file the user forgot about would lose
        their work.
        """
        if not relative.strip():
            raise ValueError("A note needs a name.")
        if not Path(relative).suffix:
            relative += ".md"
        path = self._resolve(relative)

        stem, suffix, n = path.stem, path.suffix, 1
        while path.exists():
            n += 1
            path = path.with_name("{} {}{}".format(stem, n, suffix))

        path.parent.mkdir(parents=True, exist_ok=True)
        if not text:
            text = "# {}\n\n".format(path.stem)
        path.write_text(text, encoding="utf-8")

        self.reindex_file(path)
        self.state = self._snapshot(Path(self.state.path), self.state.last_report)
        return str(path.relative_to(Path(self.state.path).resolve()))

    def create_folder(self, relative: str) -> str:
        """Create a folder in the vault. Returns its vault-relative path."""
        if not relative.strip():
            raise ValueError("A folder needs a name.")
        path = self._resolve(relative)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.relative_to(Path(self.state.path).resolve()))

    def delete_note(self, relative: str) -> bool:
        """Delete a note from disk and drop it from the index."""
        path = self._resolve(relative)
        if not path.is_file():
            return False
        path.unlink()
        self.reindex_file(path)  # absent file -> removed from the index
        self.state = self._snapshot(Path(self.state.path), self.state.last_report)
        return True

    def reveal_in_finder(self, relative: str) -> None:
        """Open the note's folder in Finder with the file selected."""
        import subprocess

        path = self._resolve(relative)
        subprocess.run(["open", "-R", str(path)], check=False)

    def folders(self) -> List[str]:
        """Existing vault folders, for the 'new note' location picker."""
        if not self.state.path:
            return []
        root = Path(self.state.path).resolve()
        from desktop.vault import SKIP_DIRECTORIES

        out = []
        for p in sorted(root.rglob("*")):
            if not p.is_dir():
                continue
            rel = p.relative_to(root)
            if any(
                part in SKIP_DIRECTORIES or part.startswith(".") for part in rel.parts
            ):
                continue
            out.append(str(rel))
        return out

    def search_only(
        self,
        question: str,
        sectors: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
    ):
        """Retrieval with no generation — free, offline, and no API key needed."""
        if not self.store:
            raise RuntimeError("Open a vault first.")
        scope = self.build_scope(sectors=sectors, doc_ids=doc_ids)
        top_k = self.settings.top_k if self.settings else 6
        # Same normalisation as ask(), so previewing retrieval matches what
        # the gate will actually score.
        return self.store.index.search(
            gate_query(question), top_k=top_k, scope=scope
        )
