"""Vault tools the model can call.

Every tool is a thin wrapper over a ``VaultController`` method, so the path
safety, atomic writes, and re-indexing already tested in ``controller.py``
apply unchanged. Nothing here reaches the filesystem directly.

Three rules shape this module:

1. **The vault is the boundary.** Every path goes through
   ``VaultController._resolve``, which refuses anything outside the vault.
   A tool cannot read ``~/.ssh/id_rsa`` no matter what the model asks for.

2. **Destructive tools are marked, not hidden.** ``write_note`` on an
   existing note and ``delete_note`` carry ``destructive = True``. The agent
   loop refuses to run those without explicit approval. Marking is done here,
   next to the implementation, so adding a tool forces the question.

3. **Errors are returned, not raised.** A tool that raises kills the loop; a
   tool that returns an error string lets the model read it and try something
   else, which is usually what a person would do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# A note read in full can be large. Truncating keeps one oversized file from
# consuming the context window and the budget along with it.
MAX_READ_CHARS = 20_000

# Search results are for orientation, not for reading in bulk.
MAX_SEARCH_HITS = 8
SEARCH_EXCERPT_CHARS = 400


@dataclass
class ToolSpec:
    """A tool definition plus the metadata the agent loop needs."""

    name: str
    description: str
    input_schema: Dict
    handler: Callable[..., str]
    # Whether running this can destroy or overwrite user data. The loop pauses
    # for approval before any destructive call.
    destructive: bool = False
    # Whether approval is required even when auto-approve is on. Shell
    # commands and writes outside the vault carry this: "stop asking me about
    # my own notes" is a reasonable preference, "run anything on my Mac
    # without telling me" is a different decision and should be made
    # separately. A tool marked here can never be silently auto-approved.
    always_confirm: bool = False

    def definition(self) -> Dict:
        """The shape the Messages API expects in `tools`."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def build_tools(controller) -> List[ToolSpec]:
    """Build the tool set bound to one open vault."""

    # --- read-only ------------------------------------------------------

    def search_vault(query: str, sector: Optional[str] = None) -> str:
        sectors = [sector] if sector else None
        hits = controller.search_only(query, sectors=sectors)
        if not hits:
            return "No matching passages. Try different wording or drop the sector filter."
        lines = []
        for i, h in enumerate(hits[:MAX_SEARCH_HITS], 1):
            path = h.chunk.metadata.get("vault_path", h.chunk.doc_name)
            excerpt = " ".join(h.chunk.text.split())[:SEARCH_EXCERPT_CHARS]
            lines.append(
                "{}. {} (sector: {})\n   {}".format(
                    i, path, h.chunk.metadata.get("sector", "?"), excerpt
                )
            )
        return "\n".join(lines)

    def read_note(path: str) -> str:
        text = controller.read_note(path)
        if len(text) > MAX_READ_CHARS:
            return (
                text[:MAX_READ_CHARS]
                + "\n\n[truncated at {} characters of {}]".format(
                    MAX_READ_CHARS, len(text)
                )
            )
        return text or "(the note is empty)"

    def list_notes(sector: Optional[str] = None) -> str:
        tree = controller.state.tree
        if sector:
            files = tree.get(sector.lower())
            if files is None:
                return "No sector named '{}'. Available: {}".format(
                    sector, ", ".join(sorted(tree))
                )
            return "\n".join(f["path"] for f in files) or "(empty)"
        out = []
        for name in sorted(tree):
            out.append("{}/ ({} notes)".format(name, len(tree[name])))
            out.extend("    " + f["path"] for f in tree[name])
        return "\n".join(out)

    def list_folders() -> str:
        folders = controller.folders()
        return "\n".join(folders) if folders else "(no subfolders; vault root only)"

    # --- mutating -------------------------------------------------------

    def create_note(path: str, content: str) -> str:
        actual = controller.create_note(path, content)
        if actual != path:
            return (
                "Created {} — note that {} already existed, so a numbered name "
                "was used instead of overwriting it.".format(actual, path)
            )
        return "Created {}".format(actual)

    def write_note(path: str, content: str) -> str:
        """Replace a note's entire contents.

        Whole-file replacement rather than a patch: the model has already read
        the note, and a partial edit protocol invites silent corruption when
        the match is ambiguous.
        """
        existed = True
        try:
            controller.read_note(path)
        except FileNotFoundError:
            existed = False
        controller.write_note(path, content)
        return "{} {} ({} characters)".format(
            "Overwrote" if existed else "Created", path, len(content)
        )

    def append_to_note(path: str, content: str) -> str:
        """Add to the end of a note without rewriting what is there.

        Safer than write_note for logs and running lists, and the common case
        for a daily note or a workout log.
        """
        try:
            existing = controller.read_note(path)
        except FileNotFoundError:
            existing = ""
        separator = "" if (not existing or existing.endswith("\n")) else "\n"
        controller.write_note(path, existing + separator + content)
        return "Appended {} characters to {}".format(len(content), path)

    def create_folder(path: str) -> str:
        rel = controller.create_folder(path)
        return (
            "Created folder {}. It will not appear in the sidebar until it "
            "contains a note.".format(rel)
        )

    def delete_note(path: str) -> str:
        if controller.delete_note(path):
            return "Deleted {}".format(path)
        return "No note at {} — nothing deleted.".format(path)

    # --- schemas --------------------------------------------------------

    path_prop = {
        "type": "string",
        "description": (
            "Vault-relative path including the folder and extension, e.g. "
            "'03_Iron/2026-10-25 Workout.md'. Never an absolute path."
        ),
    }

    return [
        ToolSpec(
            name="search_vault",
            description=(
                "Search the vault by meaning and keyword. Returns ranked "
                "passages with their note paths. Use this first to find which "
                "notes are relevant, then read_note for the full text."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for. Topic words work better than full sentences.",
                    },
                    "sector": {
                        "type": "string",
                        "description": "Optional sector to restrict to, e.g. 'academy' or 'iron'.",
                    },
                },
                "required": ["query"],
            },
            handler=search_vault,
        ),
        ToolSpec(
            name="read_note",
            description=(
                "Read a note's full current text from disk. Use after "
                "search_vault or list_notes gives you a path."
            ),
            input_schema={
                "type": "object",
                "properties": {"path": path_prop},
                "required": ["path"],
            },
            handler=read_note,
        ),
        ToolSpec(
            name="list_notes",
            description=(
                "List notes in the vault, optionally for one sector. Use to "
                "see what exists before creating something that may duplicate it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Optional sector name. Omit to list everything.",
                    }
                },
            },
            handler=list_notes,
        ),
        ToolSpec(
            name="list_folders",
            description="List the vault's folders, so a new note goes somewhere that exists.",
            input_schema={"type": "object", "properties": {}},
            handler=list_folders,
        ),
        ToolSpec(
            name="create_note",
            description=(
                "Create a new note. Never overwrites: if the path is taken, a "
                "numbered variant is used and the result says so. Prefer this "
                "over write_note when the note is new."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": path_prop,
                    "content": {
                        "type": "string",
                        "description": "Full markdown content, including a '# Title' heading.",
                    },
                },
                "required": ["path", "content"],
            },
            handler=create_note,
        ),
        ToolSpec(
            name="append_to_note",
            description=(
                "Add text to the end of an existing note, leaving the rest "
                "untouched. Prefer this over write_note when adding an entry "
                "to a log, list, or daily note."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": path_prop,
                    "content": {
                        "type": "string",
                        "description": "Markdown to append. Start with a newline or heading.",
                    },
                },
                "required": ["path", "content"],
            },
            handler=append_to_note,
        ),
        ToolSpec(
            name="write_note",
            description=(
                "Replace a note's entire contents. Read it first — this "
                "discards everything currently in the file. Use append_to_note "
                "to add without replacing."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": path_prop,
                    "content": {
                        "type": "string",
                        "description": "The complete new contents of the note.",
                    },
                },
                "required": ["path", "content"],
            },
            handler=write_note,
            destructive=True,
        ),
        ToolSpec(
            name="create_folder",
            description="Create a folder in the vault.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Vault-relative folder path, e.g. '06_Projects'.",
                    }
                },
                "required": ["path"],
            },
            handler=create_folder,
        ),
        ToolSpec(
            name="delete_note",
            description=(
                "Permanently delete a note from the vault. There is no undo "
                "inside the app. Only use when explicitly asked to delete."
            ),
            input_schema={
                "type": "object",
                "properties": {"path": path_prop},
                "required": ["path"],
            },
            handler=delete_note,
            destructive=True,
        ),
    ] + build_system_tools()


def build_system_tools() -> List[ToolSpec]:
    """Machine-wide tools: files anywhere, shell, search, and app control.

    Separate from the vault tools because the safety story is different.
    These are not fenced to a folder — the approval gate is what stands
    between a misread instruction and the filesystem, so every write and
    every command carries ``always_confirm``.
    """
    from desktop import system_tools as sys_tools

    any_path = {
        "type": "string",
        "description": "Absolute path, or one starting with ~. Shown to the user before anything runs.",
    }

    return [
        ToolSpec(
            name="run_command",
            description=(
                "Run a shell command — tests, git, package managers, build "
                "tools, anything you would type in a terminal. The user sees "
                "and approves the exact command first. Returns stdout, "
                "stderr, and the exit code. Use `cwd` instead of `cd`."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command line to run, e.g. 'pytest -q' or 'git status'.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Directory to run in. Defaults to the home directory.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds before giving up (default 120, max 600).",
                    },
                },
                "required": ["command"],
            },
            handler=sys_tools.run_command,
            destructive=True,
            always_confirm=True,
        ),
        ToolSpec(
            name="read_file",
            description=(
                "Read any text file on the machine, including source code and "
                "config outside the vault. Use read_note for vault notes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": any_path,
                    "offset": {
                        "type": "integer",
                        "description": "First line to read, 0-based. Omit to read from the start.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many lines to read. Omit for the whole file.",
                    },
                },
                "required": ["path"],
            },
            handler=sys_tools.read_file,
        ),
        ToolSpec(
            name="write_file",
            description=(
                "Create or completely replace a file anywhere on the machine. "
                "For a precise change to an existing file prefer edit_file, "
                "which cannot clobber the rest of it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": any_path,
                    "content": {"type": "string", "description": "Full file contents."},
                },
                "required": ["path", "content"],
            },
            handler=sys_tools.write_file,
            destructive=True,
            always_confirm=True,
        ),
        ToolSpec(
            name="edit_file",
            description=(
                "Replace one exact occurrence of a string in a file. Read the "
                "file first and match it character for character. Fails rather "
                "than guessing if the text is missing or appears more than once."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": any_path,
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace, with enough context to be unique.",
                    },
                    "new_text": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_text", "new_text"],
            },
            handler=sys_tools.edit_file,
            destructive=True,
            always_confirm=True,
        ),
        ToolSpec(
            name="list_directory",
            description="List the contents of any folder on the machine.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": any_path,
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Include dotfiles. Default false.",
                    },
                },
                "required": ["path"],
            },
            handler=sys_tools.list_directory,
        ),
        ToolSpec(
            name="find_files",
            description=(
                "Find files by glob pattern, e.g. '**/*.py'. Skips .git, "
                "node_modules, __pycache__, and virtualenvs."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '**/*.ts' or 'src/**/test_*.py'.",
                    },
                    "path": any_path,
                },
                "required": ["pattern"],
            },
            handler=sys_tools.find_files,
        ),
        ToolSpec(
            name="search_files",
            description=(
                "Regex search inside files across a directory — the way to "
                "find a function, class, or string in a codebase."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression."},
                    "path": any_path,
                    "file_glob": {
                        "type": "string",
                        "description": "Optional filter, e.g. '*.py'.",
                    },
                },
                "required": ["pattern"],
            },
            handler=sys_tools.search_files,
        ),
        ToolSpec(
            name="open_path",
            description=(
                "Open a file, folder, URL, or app in the default macOS "
                "application — Finder, a browser, Obsidian, and so on."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Path, URL, or application name.",
                    }
                },
                "required": ["target"],
            },
            handler=sys_tools.open_path,
        ),
    ]
