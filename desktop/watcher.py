"""Watch the vault folder and re-index notes as they change.

Editing a note in Obsidian and immediately asking about it should work. Without
a watcher the user has to remember to press Rescan, and a stale index that
silently answers from an old version of a note is worse than no index.

Two behaviours matter here:

* **Debounce.** Obsidian writes on nearly every keystroke pause, and editors
  often save via a write-to-temp-then-rename dance that emits several events
  per logical save. Re-indexing on each one would thrash. Events are therefore
  collected and flushed after a quiet period.
* **Never crash the app.** The watcher runs on a background thread. An
  exception there must not take down the UI, so the flush loop swallows and
  logs rather than propagating.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional, Set

logger = logging.getLogger(__name__)

# Quiet period before a changed file is re-indexed. Long enough to coalesce a
# burst of editor saves, short enough that the note is ready by the time the
# user switches to the chat and types a question.
DEBOUNCE_SECONDS = 1.5

try:  # watchdog is optional: the app must still run without it.
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:  # pragma: no cover
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object  # type: ignore[assignment,misc]
    Observer = None  # type: ignore[assignment]


class _DebouncedHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Collects paths and hands them to a callback after things go quiet."""

    def __init__(self, on_flush: Callable[[Set[Path]], None]):
        self._on_flush = on_flush
        self._pending: Set[Path] = set()
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    # --- watchdog hooks -------------------------------------------------

    def on_modified(self, event):
        self._queue(event)

    def on_created(self, event):
        self._queue(event)

    def on_deleted(self, event):
        self._queue(event)

    def on_moved(self, event):
        self._queue(event)
        dest = getattr(event, "dest_path", None)
        if dest:
            self._add(Path(dest))

    # --- internals ------------------------------------------------------

    def _queue(self, event) -> None:
        if getattr(event, "is_directory", False):
            return
        self._add(Path(event.src_path))

    def _add(self, path: Path) -> None:
        # Editors write swap and lock files constantly; indexing them would be
        # noise at best and an error at worst.
        if path.name.startswith(".") or path.name.startswith("~"):
            return
        with self._lock:
            self._pending.add(path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECONDS, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            batch = set(self._pending)
            self._pending.clear()
            self._timer = None
        if not batch:
            return
        try:
            self._on_flush(batch)
        except Exception:  # a watcher fault must never kill the UI
            logger.exception("Vault watcher flush failed")

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()


class VaultWatcher:
    """Start/stop a debounced recursive watch over the vault folder."""

    def __init__(self, root: Path, on_changes: Callable[[Set[Path]], None]):
        self.root = Path(root)
        self._on_changes = on_changes
        self._observer = None
        self._handler: Optional[_DebouncedHandler] = None

    @property
    def available(self) -> bool:
        return WATCHDOG_AVAILABLE

    @property
    def running(self) -> bool:
        return self._observer is not None

    def start(self) -> bool:
        """Begin watching. Returns False if unavailable or already running."""
        if not WATCHDOG_AVAILABLE or self._observer is not None:
            return False
        if not self.root.is_dir():
            return False
        try:
            self._handler = _DebouncedHandler(self._on_changes)
            observer = Observer()
            observer.schedule(self._handler, str(self.root), recursive=True)
            observer.daemon = True
            observer.start()
            self._observer = observer
            return True
        except Exception:
            # Watching a large iCloud-synced folder can fail on file-descriptor
            # limits. Manual rescan still works, so this is a degradation.
            logger.exception("Could not start vault watcher")
            self._observer = None
            return False

    def stop(self) -> None:
        if self._handler is not None:
            self._handler.cancel()
            self._handler = None
        if self._observer is None:
            return
        try:
            self._observer.stop()
            self._observer.join(timeout=3)
        except Exception:
            logger.debug("Watcher shutdown was not clean", exc_info=True)
        finally:
            self._observer = None
