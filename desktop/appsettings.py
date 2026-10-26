"""Desktop app preferences and secret storage.

The web version read everything from ``.env``. A double-clickable ``.app`` has
no shell, so there is no environment to read from — preferences live in
``~/Library/Application Support/JarvisDesktop/`` instead.

The API key is kept out of that file. It goes to the macOS Keychain via the
``security`` CLI, which ships with the OS and needs no dependency. A plaintext
key in a config file inside an iCloud-synced vault would be a real exposure:
iCloud replicates it to every device and to Apple's servers.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

APP_NAME = "JarvisDesktop"
SUPPORT_DIR = Path.home() / "Library" / "Application Support" / APP_NAME
CONFIG_PATH = SUPPORT_DIR / "settings.json"
DATA_DIR = SUPPORT_DIR / "data"

KEYCHAIN_SERVICE = "jarvis-desktop-anthropic"
KEYCHAIN_ACCOUNT = "anthropic-api-key"

MAX_RECENT_VAULTS = 8


# --- Keychain ----------------------------------------------------------


def load_api_key() -> str:
    """Read the key from the Keychain, falling back to the environment.

    The environment fallback keeps `python -m desktop.main` working the same
    way the web app did, which matters while developing.
    """
    env_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Keychain read failed: %s", exc)
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def save_api_key(key: str) -> bool:
    """Store (or clear) the key in the Keychain. Returns True on success."""
    key = (key or "").strip()
    if not key:
        return clear_api_key()
    try:
        result = subprocess.run(
            [
                "security",
                "add-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
                "-w",
                key,
                "-U",  # update if it already exists
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Keychain write failed: %s", exc)
        return False
    return result.returncode == 0


def clear_api_key() -> bool:
    try:
        subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


# --- Preferences -------------------------------------------------------


@dataclass
class AppSettings:
    """Everything the app remembers between launches."""

    vault_path: str = ""
    recent_vaults: List[str] = field(default_factory=list)

    # Mirrors of the core Settings knobs, surfaced in the settings panel so the
    # app never requires editing a .env by hand.
    model: str = "claude-opus-5"
    max_tokens: int = 2000
    # The agent writes whole files in a single tool call, so it needs far more
    # room than a chat answer. At 2000 the model hit the ceiling mid-file and
    # worked around it by splitting cli.py into three parts and `cat`-ing them
    # back together — seven rewrites of one file. Generous here is cheap:
    # max_tokens is a ceiling, and you are billed for what is produced.
    agent_max_tokens: int = 32000
    effort: str = "medium"
    top_k: int = 6
    # Measured on a real 32-note vault with the hashing embedder rather than
    # inherited from the web build; see docs/vault-tuning.md. The upstream
    # defaults (0.40 / 0.25) produced 6 false answers out of 15 off-topic
    # questions here, because the hashing embedder scores generic question
    # phrasing highly regardless of topic.
    relevance_floor: float = 0.45
    dense_rescue_floor: float = 0.35
    # Real semantic vectors rather than the zero-dependency hashing embedder.
    # Measured on the same 32-note vault, the hashing embedder could not
    # separate on- from off-topic at all (off-topic dense max 0.317 exceeded
    # in-scope median 0.247), so the dense half of the hybrid contributed
    # noise. With sentence-transformers the distributions are nearly disjoint
    # (in-scope min 0.298 vs off-topic max 0.337), which is what lets a
    # question phrased in words the notes never use still find them.
    embedding_backend: str = "sentence-transformers"
    embedding_dimensions: int = 384

    # Top-level vault folders left out of the index. Defaults exclude past AI
    # chat transcripts and generated output so the assistant cites real notes
    # rather than its own earlier answers. See desktop/vault.py.
    excluded_folders: List[str] = field(
        default_factory=lambda: ["copilot", "outputs"]
    )

    # Agent mode: Claude gets tools to search, read, and write notes rather
    # than a single retrieve-and-answer pass. On by default — it answers
    # questions as well as the Q&A path and can also act on the vault.
    agent_mode: bool = True
    # Run destructive tools (overwrite, delete) without asking first. Off by
    # default: losing a note to a misread instruction is not recoverable from
    # inside the app.
    auto_approve_edits: bool = False
    # Anthropic-hosted web search, for current docs and external facts.
    # Runs server-side and only reads public pages, so it needs no approval.
    web_search: bool = True

    # Desktop-only preferences.
    theme: str = "dark"
    font_size: int = 15
    auto_scroll: bool = True
    watch_vault: bool = True
    # Ask before every API call. Off by default; the status bar shows cost
    # after the fact instead of interrupting the flow of work.
    confirm_before_send: bool = False

    @classmethod
    def load(cls) -> "AppSettings":
        """Read settings, tolerating a missing or corrupt file."""
        if not CONFIG_PATH.exists():
            return cls()
        try:
            raw = json.loads(CONFIG_PATH.read_text("utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning("Could not read %s (%s); using defaults.", CONFIG_PATH, exc)
            return cls()
        if not isinstance(raw, dict):
            return cls()

        defaults = cls()
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        merged = {f: getattr(defaults, f) for f in known}
        for key, value in raw.items():
            if key not in known:
                continue
            expected = type(getattr(defaults, key))
            try:
                if expected is bool:
                    merged[key] = bool(value)
                elif expected is int:
                    merged[key] = int(value)
                elif expected is float:
                    merged[key] = float(value)
                elif expected is list:
                    merged[key] = [str(v) for v in value] if isinstance(value, list) else []
                else:
                    merged[key] = str(value)
            except (TypeError, ValueError):
                continue  # keep the default rather than crash on a bad value
        return cls(**merged)

    def save(self) -> None:
        SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            CONFIG_PATH.write_text(
                json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Could not save settings: %s", exc)

    def remember_vault(self, path: str) -> None:
        """Set the active vault and move it to the front of the recent list."""
        path = str(Path(path).expanduser().resolve())
        self.vault_path = path
        recents = [p for p in self.recent_vaults if p != path]
        recents.insert(0, path)
        self.recent_vaults = recents[:MAX_RECENT_VAULTS]

    def valid_recents(self) -> List[str]:
        """Recent vaults that still exist, so the launcher shows no dead rows."""
        return [p for p in self.recent_vaults if Path(p).is_dir()]

    def data_dir_for_vault(self, vault_path: Optional[str] = None) -> Path:
        """A per-vault index directory.

        Two vaults must not share a corpus, and the index must not be written
        inside the vault itself — that would make Obsidian show the index as
        notes and make the indexer see its own output.
        """
        target = vault_path or self.vault_path
        if not target:
            return DATA_DIR / "default"
        resolved = Path(target).expanduser().resolve()
        # A readable name plus a short hash keeps the folder identifiable while
        # staying unique for same-named vaults in different locations.
        import hashlib

        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in resolved.name)
        return DATA_DIR / "{}-{}".format(safe or "vault", digest)
