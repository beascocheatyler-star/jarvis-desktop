"""Index an Obsidian vault folder into the existing hybrid DocumentStore.

The web version of this project took uploads. A vault is different in three
ways that matter, and each one drives a decision here:

1. **Files change under you.** Obsidian writes a note every few seconds. A
   byte-identical re-index must be free, and a changed note must replace its
   old chunks rather than sit alongside them as a duplicate. ``DocumentStore``
   raises ``DuplicateDocumentError`` on identical bytes, which is exactly the
   "nothing to do" signal we want — we catch it and skip.

2. **Folders already carry meaning.** ``02-Academy/canvas.md`` is academic;
   ``03_Iron/`` is training. That structure is the user's own taxonomy, built
   over months. Rather than ask for metadata, we read the top-level folder and
   emit it as ``sector``, which the existing ``ScopeFilter`` consumes verbatim.
   Clicking a folder in the sidebar becomes ``{"sector": "academy"}`` with no
   translation layer in between.

3. **Frontmatter is metadata the user already wrote.** YAML keys at the top of
   a note are lifted into the same metadata dict, so ``course: CS1342`` in a
   note becomes a filterable facet for free.

Nothing here reaches into the retrieval or gate internals. The vault is turned
into ``store.add()`` calls and everything downstream is unchanged.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set

from app.core.extract import ExtractionError
from app.core.metadata import InvalidFilterError, normalize_value
from app.core.store import DocumentStore, DuplicateDocumentError

logger = logging.getLogger(__name__)

# Extensions the existing extractor understands. Anything else in the vault
# (images, canvas files, .obsidian config) is skipped rather than fed to a
# parser that would fail on it.
INDEXABLE_SUFFIXES = {
    # Notes and documents
    ".md", ".txt", ".pdf", ".docx", ".csv", ".json",
    # Source code. A vault that holds a project should let you ask about the
    # code in it, not just the notes beside it. The extractor reads these as
    # plain text, which is what BM25 and the embedder want anyway.
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".bash", ".zsh",
    ".html", ".css", ".scss", ".sql", ".yaml", ".yml", ".toml",
    ".go", ".rs", ".java", ".kt", ".swift", ".c", ".h", ".cpp", ".hpp",
    ".rb", ".php", ".lua", ".r", ".vue", ".svelte",
}

# Source-code extensions. `app.core.extract` only knows document formats, so
# these are presented to it as text — see `VaultIndexer._store_name`.
CODE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".bash", ".zsh",
    ".html", ".css", ".scss", ".sql", ".yaml", ".yml", ".toml",
    ".go", ".rs", ".java", ".kt", ".swift", ".c", ".h", ".cpp", ".hpp",
    ".rb", ".php", ".lua", ".r", ".vue", ".svelte",
}

# Formats whose text can safely have a title header prepended before
# extraction. PDF and DOCX are parsed from their byte structure, so injecting
# anything would corrupt the file; they rely on `doc_name` for citations.
# Code gets the header too — a filename is often the best description of what
# a source file does, and the header is a comment-free line of context.
TITLE_HEADER_SUFFIXES = {".md", ".txt"} | CODE_SUFFIXES

# Directories that are never notes. `.obsidian` holds editor config while
# `.smart-env` and `.copilot` are other plugins' caches.
#
# `copilot/copilot-conversations` earns its place by measurement rather than
# principle: on the vault this was built against it held 47 of 85 documents —
# transcripts of past AI chats. Indexing those makes the assistant cite its own
# earlier output as if it were a source note, which launders a previous guess
# into an apparent fact. Real notes must outrank chat logs, so the logs are out
# by default. `EXCLUDED_BY_DEFAULT` is overridable per vault; the hidden
# dot-directories above are not, since they are never user content.
SKIP_DIRECTORIES = {
    ".obsidian",
    ".smart-env",
    ".copilot",
    ".claude",
    ".git",
    ".trash",
    "node_modules",
    "__pycache__",
    ".venv",
}

# Folders skipped unless the user opts in from the settings panel.
EXCLUDED_BY_DEFAULT = {
    "copilot",          # past AI conversation transcripts
    "outputs",          # generated artifacts, including this app's own source
}

# A vault note can be enormous (a PDF textbook). Skip anything implausible as a
# note so one 200 MB file cannot stall startup.
MAX_FILE_BYTES = 25 * 1024 * 1024

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# ``normalize_metadata`` raises if a document carries more than 32 keys, and
# that exception would discard the whole dict — including the ``sector`` and
# ``vault_path`` keys the sidebar and incremental re-index depend on. Notes
# with sprawling frontmatter are real, so frontmatter is capped well below the
# core limit and the structural keys are always added last.
MAX_KEYS_PER_DOCUMENT = 32
STRUCTURAL_KEYS = 4  # sector, folder, filetype, vault_path
MAX_FRONTMATTER_KEYS = MAX_KEYS_PER_DOCUMENT - STRUCTURAL_KEYS - 2  # headroom
MAX_METADATA_VALUE = 200  # core limit is 256


class _PathMap(dict):
    """A ``{vault_path: doc_id}`` dict that normalises its keys.

    ``DocumentStore`` lowercases every metadata value it stores, so the
    ``vault_path`` written for ``01_Dashboard/today.md`` reads back as
    ``01_dashboard/today.md``. Keeping the normalisation inside the mapping
    means no caller can reintroduce the mismatch that previously caused the
    prune step to delete notes that were still on disk.
    """

    @staticmethod
    def _key(path: object) -> str:
        return normalize_value(str(path))

    def __setitem__(self, key: object, value: str) -> None:
        super().__setitem__(self._key(key), value)

    def __getitem__(self, key: object) -> str:
        return super().__getitem__(self._key(key))

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._key(key))

    def get(self, key: object, default=None):
        return super().get(self._key(key), default)

    def pop(self, key: object, default=None):
        return super().pop(self._key(key), default)


@dataclass
class IndexReport:
    """What a scan actually did, so the UI can show it instead of a spinner."""

    added: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total_seen(self) -> int:
        return self.added + self.unchanged + self.skipped + self.failed

    def summary(self) -> str:
        bits = []
        if self.added:
            bits.append("{} indexed".format(self.added))
        if self.unchanged:
            bits.append("{} unchanged".format(self.unchanged))
        if self.removed:
            bits.append("{} removed".format(self.removed))
        if self.failed:
            bits.append("{} failed".format(self.failed))
        if not bits:
            return "No notes found."
        return ", ".join(bits) + "."


def _title_words(relative: Path) -> str:
    """Filename and folders as searchable words.

    ``02-Academy/MATH3323_Exam2_Study_Guide.md`` becomes
    ``MATH3323 Exam2 Study Guide Academy``. Separators are split so BM25 sees
    individual terms, and the numeric folder prefix is dropped because "02" is
    noise rather than a word anyone searches for.
    """
    stem = re.sub(r"[_\-.]+", " ", relative.stem)
    # Split camelCase and letter/digit runs: "MATH3323" -> "MATH 3323", so the
    # bare course number matches too.
    stem = re.sub(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])", " ", stem)

    folders = []
    for part in relative.parts[:-1]:
        cleaned = re.sub(r"^[0-9]+[\s_\-]*", "", part)
        cleaned = re.sub(r"[_\-.]+", " ", cleaned).strip()
        if cleaned:
            folders.append(cleaned)

    return " ".join([stem] + folders).strip()


def add_title_header(text: str, relative: Path) -> str:
    """Prepend the note's title and location so they are retrievable.

    ``Chunk`` tokenises only ``text``; ``doc_name`` is carried for citations
    but never indexed. A note called ``canvas.md`` was therefore unfindable by
    the word "canvas" unless the body happened to repeat it — asking "what is
    on my canvas" retrieved three unrelated triage notes and was refused by
    the gate. Folding the title into the text fixes that at the source,
    without loosening the gate.

    The header is also genuinely useful context for the model, which otherwise
    sees passage text with no indication of which note it came from.

    Applied only to the first chunk's worth of text in practice, since
    chunking happens downstream — later chunks of a long note still carry the
    title through ``doc_name`` in their citation.
    """
    title = _title_words(relative)
    if not title:
        return text
    # Skip when the note already opens with a heading carrying the same words;
    # compared on tokens so "# Canvas Intel Report" counts as already titled
    # for `canvas.md` but an unrelated heading does not.
    first_line = text.lstrip().split("\n", 1)[0]
    if first_line.startswith("#"):
        heading_words = set(re.findall(r"\w+", first_line.lower()))
        title_words = set(re.findall(r"\w+", title.lower()))
        if title_words and title_words <= heading_words:
            return text
    return "# {}\n\n(Note: {})\n\n{}".format(title, relative, text)


def _relative_to(path: Path, root: Path) -> Optional[str]:
    """Vault-relative path, tolerating symlinked roots.

    On macOS ``/var`` is a symlink to ``/private/var``, and ``$TMPDIR`` sits
    under it, so a root captured as ``/var/...`` and a file discovered as
    ``/private/var/...`` describe the same location while ``relative_to``
    raises ValueError on the pair. Both sides are resolved before comparing.
    Returns None when the file genuinely lies outside the vault.
    """
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        try:
            return str(Path(path).relative_to(root))
        except ValueError:
            return None


def _sector_of(relative: Path) -> str:
    """Top-level folder as a scope value, e.g. ``02-Academy/x.md`` -> ``academy``.

    Leading digits and separators are stripped so the filter value is something
    a human would type: the user thinks "academy", not "02-Academy".
    """
    parts = relative.parts
    if len(parts) <= 1:
        return "root"
    raw = parts[0]
    cleaned = re.sub(r"^[0-9]+[\s_\-]*", "", raw).strip().lower()
    cleaned = cleaned.replace(" ", "_")
    return cleaned or raw.lower()


def _parse_frontmatter(text: str) -> Dict[str, str]:
    """Lift simple ``key: value`` YAML frontmatter into flat metadata.

    Deliberately not a YAML parser. Nested structures and lists are flattened
    to a comma-joined string or skipped; the goal is to surface the handful of
    scalar keys people actually filter on (course, status, type) without taking
    a dependency or risking an exception on malformed YAML mid-scan.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}

    out: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if len(out) >= MAX_FRONTMATTER_KEYS:
            break
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip().strip("\"'").strip("[]")
        # Structural keys are set by the indexer and must not be shadowed by a
        # note that happens to define them; overlong keys would raise in the
        # core normaliser and cost the note all of its metadata.
        if not key or not value or len(key) > 64:
            continue
        if key in {"sector", "folder", "filetype", "vault_path", "path", "vault_file"}:
            continue
        out[key] = value[:MAX_METADATA_VALUE]
    return out


def discover(
    vault_root: Path, excluded_folders: Optional[Iterable[str]] = None
) -> List[Path]:
    """Every indexable file under the vault, sorted for a stable scan order.

    ``excluded_folders`` is matched against top-level folder names, both raw
    and sector-normalised, so either ``02-Academy`` or ``academy`` excludes the
    same tree.
    """
    root = Path(vault_root).expanduser().resolve()
    excluded = {e.strip().lower() for e in (excluded_folders or ()) if e.strip()}
    found: List[Path] = []

    # Prune as we descend rather than filtering a full rglob afterwards. A
    # vault that contains a project (a .venv, a node_modules) holds orders of
    # magnitude more files than notes -- 25k vs 107 on the vault this was
    # built against -- and stat-ing all of them on an iCloud-synced folder
    # stalls the first index for minutes with nothing on screen yet.
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRECTORIES and not d.startswith(".")
        ]
        rel_dir = _relative_to(here, root)
        if rel_dir is None:
            dirnames[:] = []
            continue
        parents = Path(rel_dir).parts if rel_dir != "." else ()
        if excluded and parents:
            # _sector_of expects a path to a file and reads the sector from its
            # parent, so a bare directory would normalise to "root". Match on
            # this directory's own top component instead, same rule applied.
            top = parents[0]
            if top.lower() in excluded or _sector_of(Path(top) / "x") in excluded:
                dirnames[:] = []
                continue

        for name in filenames:
            if name.startswith("."):
                continue
            path = here / name
            if path.suffix.lower() not in INDEXABLE_SUFFIXES:
                continue
            found.append(path)
    return sorted(found)


def metadata_for(path: Path, root: Path, text: Optional[str] = None) -> Dict[str, str]:
    """Scope metadata for one note: sector, folder, extension, frontmatter."""
    relative = Path(_relative_to(path, root) or path.name)
    structural: Dict[str, str] = {
        "sector": _sector_of(relative),
        "folder": (
            str(relative.parent)[:MAX_METADATA_VALUE]
            if str(relative.parent) != "."
            else "root"
        ),
        "filetype": path.suffix.lower().lstrip("."),
    }

    if text is None and path.suffix.lower() == ".md":
        try:
            text = path.read_text("utf-8", errors="replace")
        except OSError:
            text = None

    meta: Dict[str, str] = {}
    if text:
        try:
            meta.update(_parse_frontmatter(text))
        except Exception:  # malformed frontmatter must not lose the note
            logger.debug("Unparseable frontmatter in %s", relative)

    # Structural keys last: they win any collision, and capping here guarantees
    # the result stays under the core 32-key limit no matter what the note said.
    if len(meta) > MAX_FRONTMATTER_KEYS:
        meta = dict(list(meta.items())[:MAX_FRONTMATTER_KEYS])
    meta.update(structural)
    return meta


class VaultIndexer:
    """Keeps a ``DocumentStore`` in sync with a folder on disk.

    One indexer owns one vault. ``scan()`` is idempotent: running it twice in a
    row does no work the second time, which is what makes it safe to call on a
    file-watcher event or on every app launch.
    """

    def __init__(
        self,
        store: DocumentStore,
        vault_root: Path,
        excluded_folders: Optional[Iterable[str]] = None,
    ):
        self.store = store
        self.root = Path(vault_root).expanduser().resolve()
        self.excluded_folders: Set[str] = (
            set(excluded_folders)
            if excluded_folders is not None
            else set(EXCLUDED_BY_DEFAULT)
        )
        # pypdf logs a warning per embedded font it cannot fully parse. A
        # textbook PDF emits hundreds, which would bury the real indexing
        # errors the user needs to see. The text still extracts correctly.
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        # Maps vault-relative path -> doc_id, so an edited note can have its
        # previous version deleted before the new one is added. Keys are
        # normalised on the way in because the store lowercases metadata
        # values; a raw key would never match the value read back out.
        self._by_path = _PathMap()
        self._rebuild_path_map()

    def _rebuild_path_map(self) -> None:
        """Recover the path -> doc_id map from a store loaded off disk.

        The store persists documents across restarts but keys them by name, so
        on boot we rebuild our own index from the metadata we wrote.
        """
        self._by_path = _PathMap()
        for doc in self.store.list_documents():
            vault_path = doc.metadata.get("vault_path")
            if vault_path:
                self._by_path[vault_path] = doc.doc_id

    def scan(
        self,
        progress: Optional[Callable[[int, int, str], None]] = None,
        prune: bool = True,
    ) -> IndexReport:
        """Index every note, skipping unchanged ones.

        ``progress(done, total, name)`` is called per file so the UI can show
        real movement instead of an indeterminate spinner — on a large vault
        the first scan is the longest wait in the app.
        """
        report = IndexReport()
        files = discover(self.root, self.excluded_folders)
        total = len(files)
        seen_paths: Set[str] = set()

        for i, path in enumerate(files, start=1):
            relative = _relative_to(path, self.root)
            if relative is None:
                report.skipped += 1
                continue
            seen_paths.add(relative)
            if progress:
                progress(i, total, relative)

            try:
                stat = path.stat()
                if stat.st_size == 0:
                    report.skipped += 1
                    continue
                if stat.st_size > MAX_FILE_BYTES:
                    report.skipped += 1
                    continue
                data = path.read_bytes()
            except OSError as exc:
                report.failed += 1
                report.errors.append("{}: {}".format(relative, exc))
                continue

            meta = metadata_for(path, self.root)
            meta["vault_path"] = relative

            try:
                doc = self._add_with_fallback(path, data, meta)
            except DuplicateDocumentError as dup:
                # Byte-identical to something already indexed: nothing to
                # re-embed. The path map still has to be refreshed, because the
                # existing document is the live one and the prune step below
                # treats an unmapped path as a deleted file.
                existing = getattr(dup, "existing", None)
                if existing is not None:
                    self._by_path[relative] = existing.doc_id
                report.unchanged += 1
                continue
            except ExtractionError as exc:
                report.failed += 1
                report.errors.append("{}: {}".format(relative, exc))
                continue
            except Exception as exc:  # defensive: one bad note must not end the scan
                report.failed += 1
                report.errors.append("{}: {}".format(relative, exc))
                logger.exception("Indexing %s failed", relative)
                continue

            # The note changed: drop the previous version so the corpus holds
            # one copy, not a history.
            previous = self._by_path.get(relative)
            if previous and previous != doc.doc_id:
                self.store.delete(previous)
            self._by_path[relative] = doc.doc_id
            report.added += 1

        if prune:
            report.removed = self._prune(seen_paths)
        return report

    def _prepare(self, path: Path, data: bytes, relative: str) -> bytes:
        """Fold the note's title into its text so the filename is searchable.

        Only for plain-text formats. PDF and DOCX are parsed from their byte
        structure downstream, so prepending anything would corrupt them —
        those keep the filename in ``doc_name`` for citations only.
        """
        if path.suffix.lower() not in TITLE_HEADER_SUFFIXES:
            return data
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return data  # not text after all; index it unchanged
        return add_title_header(text, Path(relative)).encode("utf-8")

    @staticmethod
    def _store_name(path: Path) -> str:
        """The filename handed to ``store.add``.

        ``app.core.extract`` dispatches on extension and raises on anything it
        does not recognise, which is every source-code type. Source is plain
        text, so it is presented as ``name.py.txt`` — the extractor takes the
        text path and the real name stays visible in citations and in the
        ``vault_path`` metadata. Changing the upstream extractor would mean
        diverging from the repo this is built on for no gain.
        """
        if path.suffix.lower() in CODE_SUFFIXES:
            return path.name + ".txt"
        return path.name

    def _add_with_fallback(self, path: Path, data: bytes, meta: Dict[str, str]):
        """Add a note, retrying with structural metadata only if the dict is rejected.

        A note is far more useful indexed with a reduced facet set than skipped
        for a frontmatter problem, and losing ``sector``/``vault_path`` would
        break both the sidebar and incremental re-indexing.
        """
        data = self._prepare(path, data, meta.get("vault_path", path.name))
        try:
            return self.store.add(self._store_name(path), data, metadata=meta)
        except InvalidFilterError as exc:
            logger.warning(
                "Metadata rejected for %s (%s); indexing with structural keys only.",
                path.name,
                exc,
            )
            minimal = {
                k: meta[k]
                for k in ("sector", "folder", "filetype", "vault_path")
                if k in meta
            }
            return self.store.add(self._store_name(path), data, metadata=minimal)

    def _prune(self, seen_paths: Set[str]) -> int:
        """Delete indexed notes whose files no longer exist on disk.

        Driven by the store's own documents rather than ``_by_path``: the map
        is a cache and any gap in it would otherwise delete a live note. A
        document is removed only on positive evidence — it records a
        ``vault_path`` and that path was not found during the scan. Documents
        without the key predate this indexer (or were added another way) and
        are left alone, since deleting data on an inference is the one mistake
        here that loses user work.
        """
        # The store lowercases every metadata value, so `01_Dashboard/x.md`
        # comes back as `01_dashboard/x.md`. Comparing raw scan paths against
        # stored ones therefore never matched for any folder with a capital
        # letter, and the prune deleted notes that were present the whole time.
        seen_normalized = {normalize_value(p) for p in seen_paths}

        removed = 0
        for doc in list(self.store.list_documents()):
            vault_path = doc.metadata.get("vault_path")
            if not vault_path or normalize_value(vault_path) in seen_normalized:
                continue
            if self.store.delete(doc.doc_id):
                removed += 1
            if self._by_path.get(vault_path) == doc.doc_id:
                self._by_path.pop(vault_path, None)
        return removed

    def reindex_file(self, path: Path) -> bool:
        """Re-index a single note after an edit. True if the corpus changed."""
        path = Path(path).resolve()
        relative = _relative_to(path, self.root)
        if relative is None:
            return False

        if not path.exists():
            doc_id = self._by_path.pop(relative, None)
            return bool(doc_id and self.store.delete(doc_id))

        if path.suffix.lower() not in INDEXABLE_SUFFIXES:
            return False

        # The watcher fires on every write in the tree, including excluded
        # folders, so the same filter scan() uses has to apply here too.
        relative_path = Path(relative)
        parents = relative_path.parts[:-1]
        if any(part in SKIP_DIRECTORIES or part.startswith(".") for part in parents):
            return False
        if self.excluded_folders and parents:
            excluded = {e.strip().lower() for e in self.excluded_folders}
            if parents[0].lower() in excluded or _sector_of(relative_path) in excluded:
                return False

        try:
            data = path.read_bytes()
        except OSError:
            return False
        if not data or len(data) > MAX_FILE_BYTES:
            return False

        meta = metadata_for(path, self.root)
        meta["vault_path"] = relative
        try:
            doc = self._add_with_fallback(path, data, meta)
        except DuplicateDocumentError:
            return False
        except Exception:
            logger.debug("Re-index of %s failed", relative, exc_info=True)
            return False

        previous = self._by_path.get(relative)
        if previous and previous != doc.doc_id:
            self.store.delete(previous)
        self._by_path[relative] = doc.doc_id
        return True

    # --- sidebar support ------------------------------------------------

    def sectors(self) -> List[str]:
        """Distinct sector values present in the corpus, for the sidebar."""
        facets = self.store.facets()
        return sorted(facets.get("sector", {}).keys())

    def tree(self) -> Dict[str, List[Dict[str, str]]]:
        """Vault files grouped by sector, mirroring the Obsidian file tree."""
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for doc in self.store.list_documents():
            sector = doc.metadata.get("sector", "root")
            grouped.setdefault(sector, []).append(
                {
                    "doc_id": doc.doc_id,
                    "name": doc.name,
                    "path": doc.metadata.get("vault_path", doc.name),
                }
            )
        for items in grouped.values():
            items.sort(key=lambda d: d["path"])
        return dict(sorted(grouped.items()))
