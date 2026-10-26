#!/usr/bin/env python3
"""Launch Jarvis.

    python jarvis.py

No server, no port, no browser. Retrieval and the scope gate run in this
process; the Anthropic API is called only when a question passes the gate.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make `app` and `desktop` importable when launched by double-click, where the
# working directory is not the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flet as ft  # noqa: E402

from desktop.ui import main  # noqa: E402


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # pypdf emits a warning for every embedded font it cannot fully parse; a
    # single textbook PDF produces hundreds and buries real errors.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    logging.getLogger("watchdog").setLevel(logging.WARNING)


if __name__ == "__main__":
    _configure_logging()
    ft.app(target=main)
