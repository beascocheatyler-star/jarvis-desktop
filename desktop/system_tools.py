"""Machine-wide tools: files anywhere, shell, search, web, and app control.

The vault tools in ``tools.py`` are deliberately fenced to one folder. These
are not. They give Jarvis the same reach a developer has at a terminal, which
is what makes it able to write and run code — and what makes the approval gate
load-bearing rather than decorative.

Safety model, in order of strength:

1. **Approval.** Every shell command and every write outside the vault stops
   and shows the user exactly what will run or change. This is the real
   control; everything below is defence in depth.
2. **A refusal list for the unrecoverable.** A small set of patterns is
   blocked outright — ``rm -rf /``, disk writes, fork bombs, piping the
   internet into a shell. These are things no legitimate request needs and
   no approval click should be able to authorise by accident.
3. **Truncation and timeouts.** Output is capped so one ``cat`` of a huge
   file cannot blow the context window, and commands time out rather than
   hanging the app.

Paths are expanded (``~``) and resolved before use so the approval dialog
shows the real target, not a relative path the user has to decode.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 30_000
MAX_READ_CHARS = 100_000
MAX_GREP_MATCHES = 100
MAX_GLOB_RESULTS = 200
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600

# Commands that cannot be undone and that no reasonable request requires.
# This is not a security boundary against a determined adversary — it is a
# guard against a model misreading an instruction, and against a user
# approving something whose consequence is not obvious from reading it.
_FORBIDDEN = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*(-rf|-fr)\s+/\s*$", "rm -rf on the filesystem root"),
    (r"\brm\s+(-[a-zA-Z]*\s+)*(-rf|-fr)\s+(/\*|~|\$HOME)\s*$", "rm -rf on your home directory"),
    (r"\bmkfs(\.|\s)", "formatting a filesystem"),
    (r"\bdd\b.*\bof=/dev/", "writing directly to a device"),
    (r">\s*/dev/(disk|sd|nvme)", "writing directly to a disk"),
    (r":\(\)\s*\{.*\}\s*;?\s*:", "a fork bomb"),
    (r"\bchmod\s+(-[a-zA-Z]+\s+)*777\s+/\s*$", "chmod 777 on the filesystem root"),
    (r"\bcurl\b[^|]*\|\s*(sudo\s+)?(ba|z|fi|)sh", "piping a download straight into a shell"),
    (r"\bwget\b[^|]*\|\s*(sudo\s+)?(ba|z|fi|)sh", "piping a download straight into a shell"),
    (r"\bsudo\s+rm\b", "sudo rm"),
    (r"\bdiskutil\s+(erase|reformat)", "erasing a disk"),
]

# Extensions worth reading as text. Anything else is reported rather than
# dumped as mojibake.
TEXT_SUFFIXES = {
    ".md", ".txt", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash", ".zsh", ".fish", ".html",
    ".css", ".scss", ".sql", ".rs", ".go", ".java", ".kt", ".swift", ".c", ".h",
    ".cpp", ".hpp", ".rb", ".php", ".pl", ".lua", ".r", ".jl", ".dart", ".vue",
    ".svelte", ".xml", ".csv", ".tsv", ".env", ".gitignore", ".dockerfile",
    ".makefile", ".gradle", ".properties", ".lock", ".log", ".diff", ".patch",
}


def check_command(command: str) -> Optional[str]:
    """Return a reason if the command is refused outright, else None."""
    collapsed = " ".join(command.split())
    for pattern, reason in _FORBIDDEN:
        if re.search(pattern, collapsed, re.IGNORECASE):
            return reason
    return None


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[truncated — {} of {} characters shown]".format(
        limit, len(text)
    )


def resolve(path: str) -> Path:
    """Expand ~ and relative segments so the approval dialog shows the truth."""
    return Path(os.path.expanduser(str(path))).resolve()


def is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES or not path.suffix:
        return True
    try:
        chunk = path.open("rb").read(2048)
    except OSError:
        return False
    return b"\x00" not in chunk


# --- implementations ----------------------------------------------------


def run_command(command: str, cwd: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a shell command and return its combined output."""
    blocked = check_command(command)
    if blocked:
        return "Refused: this looks like {}. Nothing was run.".format(blocked)

    workdir = resolve(cwd) if cwd else Path.home()
    if not workdir.is_dir():
        return "No such directory: {}".format(workdir)

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=min(int(timeout), MAX_TIMEOUT),
        )
    except subprocess.TimeoutExpired:
        return "Timed out after {}s. Nothing further was run.".format(timeout)
    except Exception as exc:
        return "Could not run the command: {}".format(exc)

    parts = []
    if proc.stdout.strip():
        parts.append(proc.stdout.rstrip())
    if proc.stderr.strip():
        parts.append("[stderr]\n" + proc.stderr.rstrip())
    if proc.returncode != 0:
        parts.append("[exit code {}]".format(proc.returncode))
    return _truncate("\n".join(parts)) if parts else "(no output; exit code 0)"


def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read any text file on the machine, optionally a line range."""
    target = resolve(path)
    if not target.exists():
        return "No such file: {}".format(target)
    if target.is_dir():
        return "{} is a directory. Use list_directory.".format(target)
    if not is_probably_text(target):
        return "{} looks binary ({}). I can't read it as text.".format(
            target.name, target.suffix or "no extension"
        )
    try:
        text = target.read_text("utf-8", errors="replace")
    except OSError as exc:
        return "Could not read {}: {}".format(target, exc)

    if offset or limit:
        lines = text.splitlines()
        start = max(0, int(offset))
        end = start + int(limit) if limit else len(lines)
        numbered = [
            "{:>6}  {}".format(i + 1, line)
            for i, line in enumerate(lines[start:end], start=start)
        ]
        return _truncate("\n".join(numbered), MAX_READ_CHARS)
    return _truncate(text, MAX_READ_CHARS)


def write_file(path: str, content: str) -> str:
    """Create or replace any file on the machine, atomically."""
    target = resolve(path)
    existed = target.exists()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".jarvis-tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        return "Could not write {}: {}".format(target, exc)
    return "{} {} ({} characters)".format(
        "Overwrote" if existed else "Created", target, len(content)
    )


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace an exact substring once — the precise alternative to rewriting.

    Refuses when the match is absent or ambiguous rather than guessing, since
    a wrong guess silently corrupts a file the model believes it fixed.
    """
    target = resolve(path)
    if not target.is_file():
        return "No such file: {}".format(target)
    try:
        text = target.read_text("utf-8", errors="replace")
    except OSError as exc:
        return "Could not read {}: {}".format(target, exc)

    count = text.count(old_text)
    if count == 0:
        return (
            "That exact text is not in {}. Read the file and match it "
            "character for character, including indentation.".format(target.name)
        )
    if count > 1:
        return (
            "That text appears {} times in {}. Include more surrounding "
            "context so it matches exactly once.".format(count, target.name)
        )

    try:
        tmp = target.with_suffix(target.suffix + ".jarvis-tmp")
        tmp.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        tmp.replace(target)
    except OSError as exc:
        return "Could not write {}: {}".format(target, exc)
    return "Edited {}".format(target)


def list_directory(path: str = "~", show_hidden: bool = False) -> str:
    target = resolve(path)
    if not target.is_dir():
        return "Not a directory: {}".format(target)
    try:
        entries = sorted(
            target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        )
    except OSError as exc:
        return "Could not list {}: {}".format(target, exc)

    lines = []
    for entry in entries:
        if entry.name.startswith(".") and not show_hidden:
            continue
        if entry.is_dir():
            lines.append("{}/".format(entry.name))
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            lines.append("{}  ({:,} bytes)".format(entry.name, size))
    return _truncate("\n".join(lines)) if lines else "(empty)"


def find_files(pattern: str, path: str = "~") -> str:
    """Glob for files by name pattern, e.g. '**/*.py'."""
    root = resolve(path)
    if not root.is_dir():
        return "Not a directory: {}".format(root)
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", ".next", "dist"}
    found = []
    try:
        for match in root.glob(pattern):
            if any(part in skip for part in match.parts):
                continue
            found.append(str(match))
            if len(found) >= MAX_GLOB_RESULTS:
                found.append("[stopped at {} results]".format(MAX_GLOB_RESULTS))
                break
    except (OSError, ValueError) as exc:
        return "Bad pattern or path: {}".format(exc)
    return "\n".join(found) if found else "No files match {} under {}".format(pattern, root)


def search_files(pattern: str, path: str = "~", file_glob: str = "") -> str:
    """Regex search file contents — ripgrep when present, Python otherwise."""
    root = resolve(path)
    if not root.exists():
        return "No such path: {}".format(root)

    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--color", "never",
               "--max-count", "5", "-m", "5"]
        if file_glob:
            cmd += ["--glob", file_glob]
        cmd += [pattern, str(root)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            out = proc.stdout.strip()
            if not out:
                return "No matches for {} under {}".format(pattern, root)
            lines = out.splitlines()[:MAX_GREP_MATCHES]
            return _truncate("\n".join(lines))
        except Exception:
            logger.debug("ripgrep failed; falling back", exc_info=True)

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return "Bad regular expression: {}".format(exc)

    skip = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    hits = []
    targets = root.rglob(file_glob or "*") if root.is_dir() else [root]
    for f in targets:
        if not f.is_file() or any(p in skip for p in f.parts):
            continue
        if not is_probably_text(f):
            continue
        try:
            for n, line in enumerate(f.read_text("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    hits.append("{}:{}: {}".format(f, n, line.strip()[:200]))
                    if len(hits) >= MAX_GREP_MATCHES:
                        return _truncate("\n".join(hits))
        except OSError:
            continue
    return "\n".join(hits) if hits else "No matches for {} under {}".format(pattern, root)


def open_path(target: str) -> str:
    """Open a file, folder, URL, or app with the macOS `open` command."""
    arg = str(target).strip()
    if not re.match(r"^(https?|file)://", arg):
        resolved = resolve(arg)
        arg = str(resolved) if resolved.exists() else str(target)
    try:
        subprocess.run(["open", arg], capture_output=True, text=True, timeout=15)
    except Exception as exc:
        return "Could not open {}: {}".format(arg, exc)
    return "Opened {}".format(arg)
