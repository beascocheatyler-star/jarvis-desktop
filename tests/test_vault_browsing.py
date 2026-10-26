"""Offline tests for browsing and editing the vault from the app."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop.appsettings import AppSettings  # noqa: E402
from desktop.controller import VaultController, gate_query  # noqa: E402


# --- gate query normalisation ------------------------------------------


def test_instruction_words_are_stripped():
    out = gate_query("Draft a short email about the MATH3323 exam").lower()
    # "draft" and "short" describe the output format, not the subject.
    for word in ("draft", "short"):
        assert word not in out.split()
    # Subject words survive. "email" is deliberately kept: a vault may well
    # contain notes about email, so stripping it could hide real matches.
    assert "math3323" in out and "exam" in out and "email" in out


def test_topic_words_survive():
    out = gate_query("What is on my canvas?")
    assert "canvas" in out.lower()


def test_all_instruction_words_falls_back_to_original():
    """A query with no topic must not become empty and refuse everything."""
    q = "please summarize briefly"
    assert gate_query(q) == q


def test_gate_query_does_not_rescue_off_topic():
    """Stripping must not turn an off-topic question into an in-scope one."""
    out = gate_query("Write me a poem about dragons").lower()
    assert "dragons" in out and "poem" in out


def test_long_question_scores_like_its_short_form(controller: VaultController):
    """Regression: natural phrasing was refused where keywords passed.

    The gate scores sqrt(bm25 x coverage); instruction words appear in no
    note, so they dragged coverage down and refused naturally worded
    questions that retrieved exactly the right passage.
    """
    import math

    def score(q: str) -> float:
        hits = controller.search_only(q)
        if not hits:
            return 0.0
        return math.sqrt(hits[0].normalized * hits[0].coverage)

    short = score("canvas")
    long = score("Can you give me a quick summary of what is on my canvas?")
    assert long >= short * 0.75, (
        "natural phrasing scored {:.2f} vs {:.2f} for keywords".format(long, short)
    )


@pytest.fixture
def controller(tmp_path: Path, monkeypatch) -> VaultController:
    vault = tmp_path / "vault"
    (vault / "02-Academy").mkdir(parents=True)
    (vault / "02-Academy" / "canvas.md").write_text("CS1342 meets Tuesday.\n")
    (vault / "03_Iron").mkdir()
    (vault / "03_Iron" / "squat.md").write_text("Squat 315.\n")
    (vault / "README.md").write_text("Root note.\n")

    settings = AppSettings()
    settings.vault_path = str(vault)
    settings.excluded_folders = []
    # Keep the index out of the vault and out of the real support directory.
    monkeypatch.setattr(
        type(settings), "data_dir_for_vault", lambda self, p=None: tmp_path / "index"
    )
    # `open_vault` calls `remember_vault()` then `save()`. Without this the
    # suite rewrites the user's real settings.json to point at a pytest temp
    # directory, so the next app launch opens an empty vault that no longer
    # exists. Tests must never touch live user state.
    monkeypatch.setattr(type(settings), "save", lambda self: None)

    c = VaultController(settings)
    c.open_vault(str(vault))
    return c


# --- reading -----------------------------------------------------------


def test_read_note_returns_disk_text(controller: VaultController):
    assert controller.read_note("02-Academy/canvas.md") == "CS1342 meets Tuesday.\n"


def test_read_missing_note_raises(controller: VaultController):
    with pytest.raises(FileNotFoundError):
        controller.read_note("nope.md")


def test_read_binary_is_refused(controller: VaultController):
    root = Path(controller.state.path)
    (root / "a.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError):
        controller.read_note("a.pdf")


# --- path safety -------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["../outside.md", "../../etc/passwd", "02-Academy/../../escape.md"]
)
def test_paths_outside_the_vault_are_refused(controller: VaultController, bad: str):
    """A relative path from stored metadata is untrusted input."""
    with pytest.raises(ValueError):
        controller.read_note(bad)
    with pytest.raises(ValueError):
        controller.write_note(bad, "x")


def test_absolute_path_escape_is_refused(controller: VaultController):
    with pytest.raises(ValueError):
        controller.read_note("/etc/passwd")


# --- writing -----------------------------------------------------------


def test_write_note_persists_and_reindexes(controller: VaultController):
    controller.write_note("03_Iron/squat.md", "Squat 405 for 1.\n")
    assert controller.read_note("03_Iron/squat.md") == "Squat 405 for 1.\n"

    hits = controller.store.index.search("405", top_k=5)
    assert any("405" in h.chunk.text for h in hits), "edit was not re-indexed"


def test_write_is_atomic_and_leaves_no_temp(controller: VaultController):
    controller.write_note("README.md", "updated\n")
    root = Path(controller.state.path)
    assert not list(root.glob("*.tmp"))


# --- creating ----------------------------------------------------------


def test_create_note_adds_extension_and_indexes(controller: VaultController):
    rel = controller.create_note("02-Academy/Lecture Notes")
    assert rel.endswith(".md")
    assert controller.read_note(rel).startswith("# Lecture Notes")

    names = {d.name for d in controller.store.list_documents()}
    assert "Lecture Notes.md" in names


def test_create_note_never_overwrites(controller: VaultController):
    first = controller.create_note("02-Academy/dup.md", "original\n")
    second = controller.create_note("02-Academy/dup.md", "second\n")
    assert first != second
    assert controller.read_note(first) == "original\n"


def test_create_folder(controller: VaultController):
    rel = controller.create_folder("06_Projects")
    assert (Path(controller.state.path) / rel).is_dir()


def test_create_folder_outside_vault_is_refused(controller: VaultController):
    with pytest.raises(ValueError):
        controller.create_folder("../evil")


# --- deleting ----------------------------------------------------------


def test_delete_note_removes_from_disk_and_index(controller: VaultController):
    assert controller.delete_note("03_Iron/squat.md") is True
    assert not (Path(controller.state.path) / "03_Iron/squat.md").exists()
    assert "squat.md" not in {d.name for d in controller.store.list_documents()}


def test_delete_missing_note_is_false(controller: VaultController):
    assert controller.delete_note("ghost.md") is False


# --- listing -----------------------------------------------------------


def test_folders_lists_vault_directories(controller: VaultController):
    folders = controller.folders()
    assert "02-Academy" in folders and "03_Iron" in folders


def test_folders_skips_hidden(controller: VaultController):
    (Path(controller.state.path) / ".obsidian").mkdir(exist_ok=True)
    assert not any(f.startswith(".") for f in controller.folders())


def test_tree_exposes_paths_for_browsing(controller: VaultController):
    tree = controller.indexer.tree()
    assert "academy" in tree
    entry = tree["academy"][0]
    assert {"doc_id", "name", "path"} <= set(entry)
    # Every listed path must be readable.
    for files in tree.values():
        for f in files:
            if f["name"].lower().endswith((".md", ".txt")):
                assert isinstance(controller.read_note(f["path"]), str)
