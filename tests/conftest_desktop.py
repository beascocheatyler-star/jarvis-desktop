"""Safety net: no test may write to the user's real application support dir.

`VaultController.open_vault` persists the vault path through
`AppSettings.save()`. A test that forgets to stub that rewrites the live
`settings.json` to a pytest temp directory, and the next app launch opens a
vault that no longer exists. This fixture makes that failure loud and
automatic rather than something each test has to remember.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import desktop.appsettings as appsettings
from app.core.embedding import reset_embedder


@pytest.fixture(autouse=True)
def isolate_app_support(tmp_path: Path, monkeypatch):
    """Redirect every settings path at the module level, for all tests."""
    support = tmp_path / "AppSupport"
    support.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(appsettings, "SUPPORT_DIR", support)
    monkeypatch.setattr(appsettings, "CONFIG_PATH", support / "settings.json")
    monkeypatch.setattr(appsettings, "DATA_DIR", support / "data")

    # The Keychain is shared machine state; never read or write it in tests.
    monkeypatch.setattr(appsettings, "load_api_key", lambda: "")
    monkeypatch.setattr(appsettings, "save_api_key", lambda key: True)

    # `get_embedder` caches one embedder for the whole process. A test that
    # builds a store with a non-default backend or dimension count leaves that
    # instance cached, and the next test silently gets the wrong vector space
    # — which showed up as an unrelated API test failing only when run after
    # the agent tests. Reset on both sides so order never matters.
    reset_embedder()
    yield
    reset_embedder()
