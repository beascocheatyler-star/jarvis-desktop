"""Offline tests for the vault indexer. No API key, no network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.embedding import build_embedder  # noqa: E402
from app.core.store import DocumentStore  # noqa: E402
from desktop.vault import (  # noqa: E402
    EXCLUDED_BY_DEFAULT,
    MAX_FRONTMATTER_KEYS,
    VaultIndexer,
    _parse_frontmatter,
    _sector_of,
    _title_words,
    add_title_header,
    discover,
    metadata_for,
)


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A miniature vault shaped like the real one.

    Lives in its own subdirectory so the index never sits inside the folder
    being scanned — that would make the indexer see its own corpus.json.
    """
    root = tmp_path / "vault"
    (root / "01_Dashboard").mkdir(parents=True)
    (root / "01_Dashboard" / "today.md").write_text(
        "---\nstatus: active\ntype: daily\n---\n\nFinish the lab report.\n"
    )
    (root / "02-Academy").mkdir()
    (root / "02-Academy" / "canvas.md").write_text(
        "# Courses\n\nCS1342 meets Tuesday. MATH3323 exam is in week 8.\n"
    )
    (root / "03_Iron").mkdir()
    (root / "03_Iron" / "squat.md").write_text("Squat 315 for 5.\n")
    (root / "copilot").mkdir()
    (root / "copilot" / "old_chat.md").write_text("A past AI conversation.\n")
    (root / ".obsidian").mkdir()
    (root / ".obsidian" / "workspace.json").write_text("{}")
    (root / "README.md").write_text("Root level note.\n")
    return root


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    return DocumentStore(
        data_dir=tmp_path / "index",
        embedder=build_embedder("hashing", 256),
    )


# --- discovery ---------------------------------------------------------


def test_discover_skips_dot_directories(vault: Path):
    names = {p.name for p in discover(vault)}
    assert "workspace.json" not in names
    assert "canvas.md" in names


def test_discover_honours_exclusions(vault: Path):
    names = {p.name for p in discover(vault, ["copilot"])}
    assert "old_chat.md" not in names
    assert "squat.md" in names


def test_discover_exclusion_matches_normalised_sector(vault: Path):
    """`academy` must exclude the folder literally named `02-Academy`."""
    names = {p.name for p in discover(vault, ["academy"])}
    assert "canvas.md" not in names


# --- metadata ----------------------------------------------------------


def test_sector_strips_numeric_prefix():
    assert _sector_of(Path("02-Academy/canvas.md")) == "academy"
    assert _sector_of(Path("01_Dashboard/x.md")) == "dashboard"
    assert _sector_of(Path("README.md")) == "root"


def test_frontmatter_is_lifted(vault: Path):
    meta = metadata_for(vault / "01_Dashboard" / "today.md", vault)
    assert meta["sector"] == "dashboard"
    assert meta["status"] == "active"
    assert meta["type"] == "daily"


def test_frontmatter_cannot_override_structural_keys(tmp_path: Path):
    note = tmp_path / "notes"
    note.mkdir()
    (note / "x.md").write_text("---\nsector: fake\nvault_path: /etc/passwd\n---\n\nhi\n")
    meta = metadata_for(note / "x.md", tmp_path)
    assert meta["sector"] == "notes"
    assert "vault_path" not in meta


def test_frontmatter_key_count_is_capped():
    body = "\n".join("k{}: v{}".format(i, i) for i in range(80))
    parsed = _parse_frontmatter("---\n{}\n---\n\ntext\n".format(body))
    assert len(parsed) <= MAX_FRONTMATTER_KEYS


# --- indexing ----------------------------------------------------------


def test_scan_indexes_notes(vault: Path, store: DocumentStore):
    report = VaultIndexer(store, vault, excluded_folders=[]).scan()
    assert report.added == 5  # 4 sector notes + README
    assert report.failed == 0
    assert len(store.list_documents()) == 5


def test_default_exclusions_skip_chat_logs(vault: Path, store: DocumentStore):
    VaultIndexer(store, vault).scan()
    names = {d.name for d in store.list_documents()}
    assert "old_chat.md" not in names
    assert "copilot" in EXCLUDED_BY_DEFAULT


def test_rescan_is_idempotent_and_preserves_corpus(vault: Path, store: DocumentStore):
    """Regression: an unchanged note must survive a second scan.

    Unchanged files raise DuplicateDocumentError and so never re-register in
    the path map. An earlier prune walked that map and deleted every note it
    had not just added, destroying 29 of 32 documents on the second scan.
    """
    indexer = VaultIndexer(store, vault, excluded_folders=[])
    first = indexer.scan()
    assert first.added == 5

    second = indexer.scan()
    assert second.added == 0
    assert second.unchanged == 5
    assert second.removed == 0, "a rescan must not delete unchanged notes"
    assert len(store.list_documents()) == 5


def test_rescan_after_restart_keeps_corpus(vault: Path, tmp_path: Path):
    """A fresh indexer over a persisted store must not prune everything."""
    data_dir = tmp_path / "persist"  # outside the vault folder
    first_store = DocumentStore(data_dir=data_dir, embedder=build_embedder("hashing", 256))
    VaultIndexer(first_store, vault, excluded_folders=[]).scan()
    assert len(first_store.list_documents()) == 5

    reopened = DocumentStore(data_dir=data_dir, embedder=build_embedder("hashing", 256))
    report = VaultIndexer(reopened, vault, excluded_folders=[]).scan()
    assert report.removed == 0
    assert len(reopened.list_documents()) == 5


def test_deleted_note_is_pruned(vault: Path, store: DocumentStore):
    indexer = VaultIndexer(store, vault, excluded_folders=[])
    indexer.scan()
    (vault / "03_Iron" / "squat.md").unlink()

    report = indexer.scan()
    assert report.removed == 1
    assert "squat.md" not in {d.name for d in store.list_documents()}


def test_edited_note_replaces_old_version(vault: Path, store: DocumentStore):
    indexer = VaultIndexer(store, vault, excluded_folders=[])
    indexer.scan()
    (vault / "03_Iron" / "squat.md").write_text("Squat 365 for 3.\n")

    indexer.scan()
    squats = [d for d in store.list_documents() if d.name == "squat.md"]
    assert len(squats) == 1, "an edit must replace, not duplicate"

    hits = store.index.search("365", top_k=3)
    assert any("365" in h.chunk.text for h in hits)


def test_reindex_file_respects_exclusions(vault: Path, store: DocumentStore):
    indexer = VaultIndexer(store, vault)  # default exclusions
    indexer.scan()
    assert indexer.reindex_file(vault / "copilot" / "old_chat.md") is False


def test_empty_and_missing_files_are_safe(vault: Path, store: DocumentStore):
    (vault / "empty.md").write_text("")
    indexer = VaultIndexer(store, vault, excluded_folders=[])
    report = indexer.scan()
    assert report.failed == 0
    assert indexer.reindex_file(vault / "nope.md") is False


# --- filename indexing -------------------------------------------------


def test_title_words_splits_separators_and_digits():
    assert _title_words(Path("02-Academy/MATH3323_Exam2_Study_Guide.md")) == (
        "MATH 3323 Exam 2 Study Guide Academy"
    )
    assert _title_words(Path("canvas.md")) == "canvas"


def test_title_header_is_prepended():
    out = add_title_header("Body text.\n", Path("02-Academy/canvas.md"))
    assert out.startswith("# canvas")
    assert "02-Academy/canvas.md" in out
    assert "Body text." in out


def test_title_header_not_duplicated_when_note_has_one():
    text = "# canvas\n\nAlready titled.\n"
    assert add_title_header(text, Path("canvas.md")) == text


def test_title_header_skipped_when_heading_contains_title():
    """`# Canvas Intel Report` already carries the word "canvas"."""
    text = "# Canvas Intel Report\n\nBody.\n"
    assert add_title_header(text, Path("canvas.md")) == text


def test_title_header_never_truncates_body():
    """Regression: a prefix-slice comparison ate the body's first character.

    The guard compared `text[:40]` against a truncated title and, when it did
    not match, returned header + text with the body's leading "#" consumed —
    producing "a# Canvas Intel Report" in the index.
    """
    body = "# Canvas Intel Report\n\n## PAST DUE\n\nItem one.\n"
    out = add_title_header(body, Path("02-Academy/notes.md"))
    assert body in out, "body must survive intact"
    assert "a#" not in out


def test_title_header_added_when_heading_is_unrelated():
    text = "# Weekly Review\n\nBody.\n"
    out = add_title_header(text, Path("canvas.md"))
    assert out.startswith("# canvas")
    assert "# Weekly Review" in out


def test_filename_is_searchable(vault: Path, store: DocumentStore):
    """Regression: a note was unfindable by its own filename.

    `canvas.md` contains no occurrence of the word "canvas" in its body, so
    asking "what is on my canvas" retrieved unrelated notes and the gate
    refused the question. Chunk tokenises only `text`, never `doc_name`.
    """
    (vault / "02-Academy" / "canvas.md").write_text(
        "CS1342 meets Tuesday. MATH3323 exam is in week 8.\n"
    )
    VaultIndexer(store, vault, excluded_folders=[]).scan()

    hits = store.index.search("what is on my canvas", top_k=5)
    assert hits, "no hits at all for a filename query"
    assert hits[0].chunk.doc_name == "canvas.md"


def test_course_code_in_filename_is_searchable(vault: Path, store: DocumentStore):
    (vault / "02-Academy" / "PHYS2326_Triage.md").write_text("Lab work pending.\n")
    VaultIndexer(store, vault, excluded_folders=[]).scan()

    hits = store.index.search("PHYS2326", top_k=5)
    assert any(h.chunk.doc_name == "PHYS2326_Triage.md" for h in hits)


def test_binary_formats_are_not_modified(tmp_path: Path):
    """A title header must never be injected into PDF/DOCX bytes."""
    root = tmp_path / "v"
    root.mkdir()
    store = DocumentStore(
        data_dir=tmp_path / "idx", embedder=build_embedder("hashing", 256)
    )
    indexer = VaultIndexer(store, root, excluded_folders=[])
    raw = b"%PDF-1.4 binary-ish bytes"
    assert indexer._prepare(root / "a.pdf", raw, "a.pdf") == raw
    assert indexer._prepare(root / "a.docx", raw, "a.docx") == raw
    # Text formats do get the header.
    out = indexer._prepare(root / "a.md", b"hello", "a.md")
    assert out.startswith(b"# a")


# --- scoping -----------------------------------------------------------


def test_sector_scope_filters_results(vault: Path, store: DocumentStore):
    from app.core.metadata import ScopeFilter

    VaultIndexer(store, vault, excluded_folders=[]).scan()
    scope = ScopeFilter.parse({"sector": ["academy"]})
    hits = store.index.search("exam", top_k=5, scope=scope)
    assert hits
    assert all(h.chunk.metadata.get("sector") == "academy" for h in hits)


def test_tree_groups_by_sector(vault: Path, store: DocumentStore):
    indexer = VaultIndexer(store, vault, excluded_folders=[])
    indexer.scan()
    tree = indexer.tree()
    assert "academy" in tree and "iron" in tree
    assert [f["name"] for f in tree["academy"]] == ["canvas.md"]


def test_tree_omits_excluded_sectors(vault: Path, store: DocumentStore):
    indexer = VaultIndexer(store, vault)  # default exclusions apply
    indexer.scan()
    assert "copilot" not in indexer.tree()
