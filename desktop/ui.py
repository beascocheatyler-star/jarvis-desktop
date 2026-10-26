"""Flet UI: an Obsidian-shaped window over the hybrid scoped RAG core.

Layout mirrors Obsidian because that is the muscle memory already in place:

    ┌──────────────┬────────────────────────────────┐
    │ vault name   │  header: scope + status        │
    │ sector list  ├────────────────────────────────┤
    │ (file tree)  │  chat transcript (scrollable)  │
    │              ├────────────────────────────────┤
    │ settings     │  composer                      │
    └──────────────┴────────────────────────────────┘

Design decisions that exist specifically to reduce friction:

* **The composer is always focused.** Launch, send, switch scope — the cursor
  returns to the input every time. Nothing is ever one unexplained click away.
* **Status is a sentence, not a spinner.** "Searching 32 notes…" then
  "Writing…" tells you which stage is slow. An indeterminate spinner just
  reads as "frozen".
* **The transcript auto-scrolls but never fights you.** ``auto_scroll`` pins to
  the newest message; scrolling up to re-read is still possible and a
  jump-to-latest button returns.
* **Refusals look different from answers.** A gate refusal is a normal,
  zero-cost outcome, so it gets its own muted style rather than an error.
* **Nothing blocks the UI thread.** Indexing and generation run on workers.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set

import flet as ft

from desktop.appsettings import AppSettings, load_api_key, save_api_key
from desktop.controller import VaultController
from desktop.watcher import VaultWatcher

logger = logging.getLogger(__name__)

# Palette. Dark by default: this is a tool for long sessions.
BG = "#1e1e1e"
PANEL = "#252526"
BORDER = "#333333"
ACCENT = "#7c6ff0"
TEXT = "#e8e8e8"
MUTED = "#9a9a9a"
USER_BUBBLE = "#2d2b55"
REFUSAL = "#3a3320"

SECTOR_ICONS = {
    "dashboard": ft.Icons.DASHBOARD_OUTLINED,
    "academy": ft.Icons.SCHOOL_OUTLINED,
    "iron": ft.Icons.FITNESS_CENTER_OUTLINED,
    "the_man": ft.Icons.PERSON_OUTLINE,
    "philosophy": ft.Icons.MENU_BOOK_OUTLINED,
    "root": ft.Icons.FOLDER_OUTLINED,
}


def sector_icon(name: str) -> str:
    return SECTOR_ICONS.get(name, ft.Icons.FOLDER_OUTLINED)


def pretty_sector(name: str) -> str:
    return name.replace("_", " ").title()


class JarvisApp:
    """The whole window. One instance per launch."""

    def __init__(self, page: ft.Page):
        self.page = page
        self.settings = AppSettings.load()
        self.controller = VaultController(self.settings)
        self.watcher: Optional[VaultWatcher] = None

        self.selected_sectors: Set[str] = set()
        # Sectors expanded in the file tree. Browsing and search-scoping are
        # deliberately separate: expanding a folder to look inside it should
        # not silently change what the assistant can see.
        self.expanded_sectors: Set[str] = set()
        self.open_note_path: Optional[str] = None
        self.editing = False
        self.busy = False

        self._build()
        self._boot()

    # --- construction ---------------------------------------------------

    def _build(self) -> None:
        page = self.page
        page.title = "Jarvis"
        page.theme_mode = (
            ft.ThemeMode.DARK if self.settings.theme == "dark" else ft.ThemeMode.LIGHT
        )
        page.bgcolor = BG
        page.padding = 0
        page.spacing = 0
        page.window.width = 1180
        page.window.height = 820
        page.window.min_width = 780
        page.window.min_height = 560
        page.on_keyboard_event = self._on_key

        # --- transcript -------------------------------------------------
        # auto_scroll keeps the newest message visible; the ListView itself
        # provides the trackpad/scroll-wheel scrollback.
        self.transcript = ft.ListView(
            expand=True,
            spacing=14,
            padding=ft.padding.symmetric(horizontal=28, vertical=20),
            auto_scroll=self.settings.auto_scroll,
            on_scroll=self._on_transcript_scroll,
        )

        self.jump_button = ft.FloatingActionButton(
            icon=ft.Icons.ARROW_DOWNWARD,
            bgcolor=ACCENT,
            mini=True,
            visible=False,
            tooltip="Jump to latest",
            on_click=self._jump_to_latest,
        )

        # --- composer -----------------------------------------------------
        self.composer = ft.TextField(
            hint_text="Ask your vault…   (Enter to send, Shift+Enter for a new line)",
            multiline=True,
            min_lines=1,
            max_lines=6,
            shift_enter=True,  # Enter sends; Shift+Enter inserts a newline
            autofocus=True,
            filled=True,
            bgcolor=PANEL,
            border=ft.InputBorder.NONE,
            border_radius=12,
            text_size=self.settings.font_size,
            color=TEXT,
            content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
            on_submit=self._on_send,
            expand=True,
        )

        self.send_button = ft.IconButton(
            icon=ft.Icons.ARROW_UPWARD_ROUNDED,
            icon_color="#ffffff",
            bgcolor=ACCENT,
            tooltip="Send  (Enter)",
            on_click=self._on_send,
        )

        # Agent mode lets Claude use tools — search, read, and write notes.
        # On by default: it handles questions as well as the Q&A path and can
        # also act. Ask mode stays available because it is cheaper (one call,
        # no tool round-trips) and cannot modify anything.
        self.agent_switch = ft.Switch(
            value=self.settings.agent_mode,
            active_color=ACCENT,
            tooltip="Agent mode: Jarvis can search, read, and write notes",
            on_change=self._toggle_agent_mode,
            scale=0.8,
        )
        self.agent_label = ft.Text(
            "Agent" if self.settings.agent_mode else "Ask",
            size=11,
            color=ACCENT if self.settings.agent_mode else MUTED,
        )

        # --- sidebar ------------------------------------------------------
        self.vault_label = ft.Text(
            "No vault", size=15, weight=ft.FontWeight.W_600, color=TEXT
        )
        self.vault_stats = ft.Text("", size=11, color=MUTED)
        self.sector_list = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO, expand=True)

        self.scope_label = ft.Text(
            "Whole vault", size=12, color=MUTED, italic=True
        )
        self.status_text = ft.Text("", size=12, color=MUTED)
        self.progress = ft.ProgressBar(
            visible=False, color=ACCENT, bgcolor=BORDER, height=2
        )

        sidebar = ft.Container(
            width=248,
            bgcolor=PANEL,
            padding=ft.padding.only(left=14, right=10, top=18, bottom=14),
            content=ft.Column(
                spacing=10,
                controls=[
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.HEXAGON_OUTLINED, color=ACCENT, size=18),
                            ft.Column(
                                [self.vault_label, self.vault_stats],
                                spacing=0,
                                expand=True,
                            ),
                        ],
                        spacing=8,
                    ),
                    ft.Divider(height=1, color=BORDER),
                    ft.Row(
                        [
                            ft.Text(
                                "SECTORS", size=10, color=MUTED,
                                weight=ft.FontWeight.BOLD,
                            ),
                            ft.Container(expand=True),
                            ft.TextButton(
                                "All",
                                on_click=self._clear_scope,
                                style=ft.ButtonStyle(color=ACCENT),
                            ),
                        ],
                        spacing=4,
                    ),
                    self.sector_list,
                    ft.Divider(height=1, color=BORDER),
                    ft.Row(
                        [
                            ft.IconButton(
                                ft.Icons.NOTE_ADD_OUTLINED,
                                tooltip="New note  (⌘N)",
                                icon_color=MUTED,
                                icon_size=19,
                                on_click=self._new_note,
                            ),
                            ft.IconButton(
                                ft.Icons.CREATE_NEW_FOLDER_OUTLINED,
                                tooltip="New folder  (⇧⌘N)",
                                icon_color=MUTED,
                                icon_size=19,
                                on_click=self._new_folder,
                            ),
                            ft.IconButton(
                                ft.Icons.ADD_COMMENT_OUTLINED,
                                tooltip="New chat  (⌘T)",
                                icon_color=MUTED,
                                icon_size=19,
                                on_click=self._new_chat,
                            ),
                            ft.IconButton(
                                ft.Icons.REFRESH,
                                tooltip="Rescan vault  (⌘R)",
                                icon_color=MUTED,
                                icon_size=19,
                                on_click=self._on_rescan,
                            ),
                            ft.IconButton(
                                ft.Icons.FOLDER_OPEN_OUTLINED,
                                tooltip="Open a different vault",
                                icon_color=MUTED,
                                icon_size=19,
                                on_click=self._pick_vault,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                    ),
                    # Settings gets its own labelled row rather than a sixth
                    # icon. Six IconButtons need roughly 288px and the sidebar
                    # is 248, so the last one — this one — rendered off the
                    # edge of the window and was simply unreachable. It is
                    # also the control people look for when something is
                    # wrong (a dead API key), which is the worst time to be
                    # hunting for an unlabelled glyph.
                    ft.Container(
                        border_radius=8,
                        padding=ft.padding.symmetric(horizontal=10, vertical=8),
                        on_click=self._open_settings,
                        tooltip="Settings  (⌘,)",
                        content=ft.Row(
                            spacing=9,
                            controls=[
                                ft.Icon(
                                    ft.Icons.SETTINGS_OUTLINED, size=16, color=MUTED
                                ),
                                ft.Text(
                                    "Settings", size=12, color=TEXT, expand=True
                                ),
                                ft.Text("⌘,", size=11, color=MUTED),
                            ],
                        ),
                    ),
                ],
            ),
        )

        header = ft.Container(
            bgcolor=PANEL,
            padding=ft.padding.symmetric(horizontal=20, vertical=10),
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.FILTER_ALT_OUTLINED, size=15, color=MUTED),
                    self.scope_label,
                    ft.Container(expand=True),
                    self.status_text,
                ],
                spacing=8,
            ),
        )

        composer_bar = ft.Container(
            bgcolor=BG,
            padding=ft.padding.only(left=24, right=24, top=8, bottom=14),
            content=ft.Column(
                spacing=4,
                controls=[
                    ft.Row(
                        [self.composer, self.send_button],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.END,
                    ),
                    ft.Row(
                        spacing=4,
                        controls=[
                            self.agent_switch,
                            self.agent_label,
                            ft.Container(expand=True),
                        ],
                    ),
                ],
            ),
        )

        self.chat_view = ft.Column(
            expand=True,
            spacing=0,
            controls=[
                ft.Container(
                    expand=True,
                    content=ft.Stack(
                        [
                            self.transcript,
                            ft.Container(
                                content=self.jump_button,
                                alignment=ft.alignment.bottom_right,
                                padding=ft.padding.only(right=18, bottom=14),
                            ),
                        ]
                    ),
                ),
                composer_bar,
            ],
        )

        # --- note view ----------------------------------------------------
        # Rendered markdown by default; the same text in a TextField when
        # editing. Reading is the common case, so it is the default.
        self.note_title = ft.Text("", size=15, weight=ft.FontWeight.W_600, color=TEXT)
        self.note_path_label = ft.Text("", size=11, color=MUTED)
        self.note_body = ft.Column(
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            spacing=0,
        )
        self.note_editor = ft.TextField(
            multiline=True,
            min_lines=20,
            max_lines=None,
            visible=False,
            filled=True,
            bgcolor=PANEL,
            border=ft.InputBorder.NONE,
            color=TEXT,
            text_size=self.settings.font_size,
            content_padding=ft.padding.all(16),
        )
        self.edit_button = ft.IconButton(
            ft.Icons.EDIT_OUTLINED, tooltip="Edit note", icon_color=MUTED,
            icon_size=18, on_click=self._toggle_edit,
        )
        self.save_button = ft.IconButton(
            ft.Icons.CHECK, tooltip="Save  (⌘S)", icon_color=ACCENT,
            icon_size=18, visible=False, on_click=self._save_note,
        )

        self.note_view = ft.Column(
            expand=True,
            spacing=0,
            visible=False,
            controls=[
                ft.Container(
                    bgcolor=PANEL,
                    padding=ft.padding.symmetric(horizontal=18, vertical=10),
                    content=ft.Row(
                        spacing=8,
                        controls=[
                            ft.IconButton(
                                ft.Icons.ARROW_BACK, tooltip="Back to chat",
                                icon_color=MUTED, icon_size=18,
                                on_click=self._close_note,
                            ),
                            ft.Column(
                                [self.note_title, self.note_path_label],
                                spacing=0, expand=True,
                            ),
                            self.edit_button,
                            self.save_button,
                            ft.IconButton(
                                ft.Icons.FOLDER_OPEN_OUTLINED,
                                tooltip="Reveal in Finder", icon_color=MUTED,
                                icon_size=18, on_click=self._reveal_note,
                            ),
                            ft.IconButton(
                                ft.Icons.DELETE_OUTLINE, tooltip="Delete note",
                                icon_color=MUTED, icon_size=18,
                                on_click=self._confirm_delete_note,
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    expand=True,
                    padding=ft.padding.symmetric(horizontal=24, vertical=16),
                    content=ft.Stack([self.note_body, self.note_editor]),
                ),
            ],
        )

        main = ft.Column(
            expand=True,
            spacing=0,
            controls=[
                header,
                self.progress,
                ft.Container(
                    expand=True,
                    content=ft.Stack([self.chat_view, self.note_view]),
                ),
            ],
        )

        self.file_picker = ft.FilePicker(on_result=self._on_vault_picked)
        page.overlay.append(self.file_picker)

        page.add(
            ft.Row(
                [sidebar, ft.VerticalDivider(width=1, color=BORDER), main],
                spacing=0,
                expand=True,
            )
        )

    # --- startup ---------------------------------------------------------

    def _boot(self) -> None:
        """Open the last vault automatically; never show a setup wall."""
        target = self.settings.vault_path
        if target and Path(target).is_dir():
            self._open_vault_async(target)
        else:
            self._system_message(
                "### Welcome\n\n"
                "Open a folder to use as your vault — your notes stay where "
                "they are and nothing is uploaded anywhere.\n\n"
                "Retrieval and the scope gate run locally. An Anthropic API "
                "key is only needed to *write* answers; searching is free."
            )
            self._set_status("Choose a vault to begin")

    # --- vault -----------------------------------------------------------

    def _pick_vault(self, _=None) -> None:
        self.file_picker.get_directory_path(dialog_title="Choose your vault folder")

    def _on_vault_picked(self, e: ft.FilePickerResultEvent) -> None:
        if e.path:
            self._open_vault_async(e.path)

    def _open_vault_async(self, path: str) -> None:
        self._set_busy(True, "Indexing…")

        def work():
            try:
                def progress(done: int, total: int, name: str):
                    if done % 5 == 0 or done == total:
                        self._set_status("Indexing {}/{} · {}".format(done, total, name))

                state = self.controller.open_vault(path, progress=progress)
            except Exception as exc:
                logger.exception("Opening vault failed")
                self._set_busy(False, "")
                self._error("Could not open that folder: {}".format(exc))
                return

            self.settings = self.controller.app_settings
            self._refresh_sidebar()
            self._start_watcher()
            self._set_busy(False, "")

            report = state.last_report
            detail = report.summary() if report else ""
            self._system_message(
                "**{}** is ready — {} notes, {} passages indexed.\n\n{}".format(
                    state.name, state.document_count, state.chunk_count, detail
                )
                + self._key_hint()
            )
            self._focus_composer()

        threading.Thread(target=work, daemon=True).start()

    def _on_rescan(self, _=None) -> None:
        if not self.controller.is_open or self.busy:
            return
        self._set_busy(True, "Rescanning…")

        def work():
            try:
                state = self.controller.rescan()
            except Exception as exc:
                self._set_busy(False, "")
                self._error("Rescan failed: {}".format(exc))
                return
            self._refresh_sidebar()
            self._set_busy(False, "")
            if state.last_report:
                self._set_status(state.last_report.summary())

        threading.Thread(target=work, daemon=True).start()

    def _start_watcher(self) -> None:
        if self.watcher is not None:
            self.watcher.stop()
            self.watcher = None
        if not self.settings.watch_vault or not self.controller.state.path:
            return

        def on_changes(paths: Set[Path]) -> None:
            changed = 0
            for p in paths:
                try:
                    if self.controller.reindex_file(p):
                        changed += 1
                except Exception:
                    logger.debug("Re-index failed for %s", p, exc_info=True)
            if changed:
                self._refresh_sidebar()
                self._set_status("{} note(s) updated".format(changed))
                # If the note on screen was edited in Obsidian, show the new
                # text — unless it is being edited here, where reloading
                # would discard what the user is typing.
                if self.open_note_path and not self.editing:
                    root = Path(self.controller.state.path).resolve()
                    changed_rel = {
                        str(p.resolve()).lower() for p in paths
                    }
                    open_abs = str((root / self.open_note_path).resolve()).lower()
                    if open_abs in changed_rel:
                        self._open_note(self.open_note_path)

        self.watcher = VaultWatcher(Path(self.controller.state.path), on_changes)
        if not self.watcher.start():
            logger.info("Vault watching unavailable; manual rescan still works.")

    # --- sidebar ----------------------------------------------------------

    def _refresh_sidebar(self) -> None:
        """Rebuild the file tree.

        Two distinct actions share each sector row, so they get separate hit
        targets rather than one overloaded click:

        * the **arrow and name** expand the sector to reveal its notes
        * the **circle on the right** toggles whether the sector limits search

        Conflating them was the original mistake — clicking a sector silently
        changed the search scope and looked like a file browser that refused
        to open.
        """
        state = self.controller.state
        self.vault_label.value = state.name or "No vault"
        self.vault_stats.value = "{} notes · {} passages".format(
            state.document_count, state.chunk_count
        )

        rows: List[ft.Control] = []
        for sector in state.sectors:
            files = state.tree.get(sector, [])
            scoped = sector in self.selected_sectors
            expanded = sector in self.expanded_sectors

            rows.append(
                ft.Container(
                    border_radius=8,
                    bgcolor="#2f2d4a" if scoped else None,
                    padding=ft.padding.only(left=4, right=6, top=5, bottom=5),
                    content=ft.Row(
                        spacing=4,
                        controls=[
                            ft.Icon(
                                ft.Icons.KEYBOARD_ARROW_DOWN
                                if expanded
                                else ft.Icons.KEYBOARD_ARROW_RIGHT,
                                size=16,
                                color=MUTED,
                            ),
                            ft.Icon(sector_icon(sector), size=15, color=MUTED),
                            ft.Container(
                                expand=True,
                                on_click=lambda e, s=sector: self._toggle_expand(s),
                                content=ft.Text(
                                    pretty_sector(sector), size=13, color=TEXT
                                ),
                            ),
                            ft.Text(str(len(files)), size=11, color=MUTED),
                            ft.IconButton(
                                icon=ft.Icons.CHECK_CIRCLE
                                if scoped
                                else ft.Icons.RADIO_BUTTON_UNCHECKED,
                                icon_size=16,
                                icon_color=ACCENT if scoped else "#555555",
                                tooltip="Only search this sector"
                                if not scoped
                                else "Stop limiting search to this sector",
                                on_click=lambda e, s=sector: self._toggle_sector(s),
                            ),
                        ],
                    ),
                )
            )

            if not expanded:
                continue

            for f in files:
                is_open = self.open_note_path == f["path"]
                rows.append(
                    ft.Container(
                        border_radius=6,
                        bgcolor=ACCENT if is_open else None,
                        padding=ft.padding.only(left=38, right=8, top=5, bottom=5),
                        on_click=lambda e, p=f["path"]: self._open_note(p),
                        content=ft.Row(
                            spacing=7,
                            controls=[
                                ft.Icon(
                                    ft.Icons.DESCRIPTION_OUTLINED,
                                    size=13,
                                    color="#ffffff" if is_open else MUTED,
                                ),
                                ft.Text(
                                    f["name"],
                                    size=12,
                                    color="#ffffff" if is_open else "#c8c8c8",
                                    expand=True,
                                    no_wrap=True,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                    tooltip=f["path"],
                                ),
                            ],
                        ),
                    )
                )

        self.sector_list.controls = rows
        self._refresh_scope_label()
        self._safe_update()

    def _toggle_expand(self, sector: str) -> None:
        if sector in self.expanded_sectors:
            self.expanded_sectors.discard(sector)
        else:
            self.expanded_sectors.add(sector)
        self._refresh_sidebar()

    # --- note viewing and editing ---------------------------------------

    def _open_note(self, relative: str) -> None:
        try:
            text = self.controller.read_note(relative)
        except ValueError as exc:
            # Binary formats (PDF, DOCX) are indexed but not viewable here.
            self._error(str(exc))
            return
        except (FileNotFoundError, OSError) as exc:
            self._error("Could not open that note: {}".format(exc))
            return

        self.open_note_path = relative
        self.editing = False
        self.note_title.value = Path(relative).name
        self.note_path_label.value = relative
        self.note_body.controls = [self._markdown(text)]
        self.note_editor.value = text
        self.note_editor.visible = False
        self.note_body.visible = True
        self.save_button.visible = False
        self.edit_button.icon = ft.Icons.EDIT_OUTLINED
        self.note_view.visible = True
        self.chat_view.visible = False
        self._refresh_sidebar()

    def _close_note(self, _=None) -> None:
        self.open_note_path = None
        self.editing = False
        self.note_view.visible = False
        self.chat_view.visible = True
        self._refresh_sidebar()
        self._focus_composer()

    def _toggle_edit(self, _=None) -> None:
        if not self.open_note_path:
            return
        self.editing = not self.editing
        self.note_editor.visible = self.editing
        self.note_body.visible = not self.editing
        self.save_button.visible = self.editing
        self.edit_button.icon = (
            ft.Icons.VISIBILITY_OUTLINED if self.editing else ft.Icons.EDIT_OUTLINED
        )
        self.edit_button.tooltip = "Preview" if self.editing else "Edit note"
        if self.editing:
            try:
                self.note_editor.focus()
            except Exception:
                # focus() updates the control, which raises if it is not
                # mounted yet. Not being able to focus is never worth a crash.
                logger.debug("Could not focus the editor", exc_info=True)
        self._safe_update()

    def _save_note(self, _=None) -> None:
        if not self.open_note_path:
            return
        text = self.note_editor.value or ""
        path = self.open_note_path

        def work():
            try:
                self.controller.write_note(path, text)
            except Exception as exc:
                logger.exception("Save failed")
                self._error("Could not save: {}".format(exc))
                return
            self.note_body.controls = [self._markdown(text)]
            self.editing = False
            self.note_editor.visible = False
            self.note_body.visible = True
            self.save_button.visible = False
            self.edit_button.icon = ft.Icons.EDIT_OUTLINED
            self._refresh_sidebar()
            self._set_status("Saved and re-indexed {}".format(Path(path).name))

        threading.Thread(target=work, daemon=True).start()

    def _reveal_note(self, _=None) -> None:
        if self.open_note_path:
            self.controller.reveal_in_finder(self.open_note_path)

    def _confirm_delete_note(self, _=None) -> None:
        """Deleting a file is irreversible, so it always asks first."""
        if not self.open_note_path:
            return
        path = self.open_note_path

        def do_delete(_):
            self.page.close(dialog)
            try:
                self.controller.delete_note(path)
            except Exception as exc:
                self._error("Could not delete: {}".format(exc))
                return
            self._close_note()
            self._set_status("Deleted {}".format(Path(path).name))

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Delete this note?"),
            content=ft.Text(
                "{}\n\nThis removes the file from your vault. It cannot be "
                "undone from inside Jarvis.".format(path)
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self.page.close(dialog)),
                ft.FilledButton(
                    "Delete",
                    on_click=do_delete,
                    style=ft.ButtonStyle(bgcolor="#8a2f2f", color="#ffffff"),
                ),
            ],
        )
        self.page.open(dialog)

    # --- creating notes and folders --------------------------------------

    def _new_note(self, _=None) -> None:
        if not self.controller.is_open:
            self._error("Open a vault first.")
            return

        folders = [""] + self.controller.folders()
        name = ft.TextField(label="Note name", hint_text="Lecture 7", autofocus=True)
        where = ft.Dropdown(
            label="Folder",
            value=sorted(self.selected_sectors)[0] if self.selected_sectors else "",
            options=[
                ft.dropdown.Option(key=f, text=f or "(vault root)") for f in folders
            ],
        )

        def create(_):
            raw = (name.value or "").strip()
            if not raw:
                self._error("Give the note a name.")
                return
            target = "{}/{}".format(where.value, raw) if where.value else raw
            self.page.close(dialog)

            def work():
                try:
                    rel = self.controller.create_note(target)
                except Exception as exc:
                    self._error("Could not create note: {}".format(exc))
                    return
                sector = self._sector_for_path(rel)
                if sector:
                    self.expanded_sectors.add(sector)
                self._refresh_sidebar()
                self._open_note(rel)
                self._toggle_edit()
                self._set_status("Created {}".format(rel))

            threading.Thread(target=work, daemon=True).start()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("New note"),
            content=ft.Container(
                width=400,
                content=ft.Column([name, where], tight=True, spacing=14),
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self.page.close(dialog)),
                ft.FilledButton("Create", on_click=create),
            ],
        )
        self.page.open(dialog)

    def _new_folder(self, _=None) -> None:
        if not self.controller.is_open:
            self._error("Open a vault first.")
            return

        parents = [""] + self.controller.folders()
        name = ft.TextField(label="Folder name", hint_text="06_Projects", autofocus=True)
        where = ft.Dropdown(
            label="Inside",
            value="",
            options=[
                ft.dropdown.Option(key=f, text=f or "(vault root)") for f in parents
            ],
        )

        def create(_):
            raw = (name.value or "").strip()
            if not raw:
                self._error("Give the folder a name.")
                return
            target = "{}/{}".format(where.value, raw) if where.value else raw
            self.page.close(dialog)
            try:
                rel = self.controller.create_folder(target)
            except Exception as exc:
                self._error("Could not create folder: {}".format(exc))
                return
            # An empty folder holds no notes, so it cannot appear in the tree
            # until something is in it — say so rather than look broken.
            self._set_status(
                "Created {} — add a note to it and it will appear".format(rel)
            )

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("New folder"),
            content=ft.Container(
                width=400,
                content=ft.Column([name, where], tight=True, spacing=14),
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self.page.close(dialog)),
                ft.FilledButton("Create", on_click=create),
            ],
        )
        self.page.open(dialog)

    def _sector_for_path(self, relative: str) -> Optional[str]:
        """Which sector a newly created note landed in, so it can be revealed."""
        for sector, files in self.controller.state.tree.items():
            if any(f["path"].lower() == relative.lower() for f in files):
                return sector
        return None

    def _toggle_sector(self, sector: str) -> None:
        if sector in self.selected_sectors:
            self.selected_sectors.discard(sector)
        else:
            self.selected_sectors.add(sector)
        self._refresh_sidebar()
        self._focus_composer()

    def _clear_scope(self, _=None) -> None:
        self.selected_sectors.clear()
        self._refresh_sidebar()
        self._focus_composer()

    def _refresh_scope_label(self) -> None:
        if not self.selected_sectors:
            self.scope_label.value = "Whole vault"
        else:
            names = ", ".join(sorted(pretty_sector(s) for s in self.selected_sectors))
            self.scope_label.value = "Scoped to {}".format(names)

    # --- chat --------------------------------------------------------------

    def _on_send(self, _=None) -> None:
        text = (self.composer.value or "").strip()
        if not text or self.busy:
            return
        if not self.controller.is_open:
            self._error("Open a vault first.")
            return

        self.composer.value = ""
        self._safe_update()
        self._add_user_message(text)

        if self.settings.agent_mode:
            self._run_agent(text)
            return

        self._set_busy(True, "Searching {} notes…".format(
            self.controller.state.document_count
        ))
        sectors = sorted(self.selected_sectors) or None

        def work():
            try:
                answer = self.controller.ask(text, sectors=sectors)
            except Exception as exc:
                logger.exception("Answer failed")
                self._set_busy(False, "")
                self._error(str(exc))
                self._focus_composer()
                return
            self._set_busy(False, "")
            self._add_answer(answer)
            self._focus_composer()

        threading.Thread(target=work, daemon=True).start()

    # --- agent mode -------------------------------------------------------

    def _run_agent(self, text: str) -> None:
        """Run the tool-using agent, streaming its steps into the transcript."""
        self._set_busy(True, "Working…")
        activity = ft.Column(spacing=3, tight=True)
        card = ft.Container(
            bgcolor=PANEL,
            border_radius=12,
            padding=ft.padding.symmetric(horizontal=16, vertical=12),
            content=activity,
            expand=True,
        )
        self.transcript.controls.append(
            ft.Row([card], alignment=ft.MainAxisAlignment.START)
        )
        self._safe_update()

        def on_step(step):
            # Tool calls are the interesting trace; the model's interim text
            # is shown only as the final answer, to avoid a wall of drafts.
            if step.kind != "tool":
                return
            activity.controls.append(
                ft.Row(
                    spacing=7,
                    controls=[
                        ft.Icon(ft.Icons.CHEVRON_RIGHT, size=13, color=ACCENT),
                        ft.Text(step.label, size=12, color=MUTED, expand=True),
                    ],
                )
            )
            self._set_status(step.label)
            self._safe_update()

        def work():
            try:
                result = self.controller.act(
                    text,
                    on_step=on_step,
                    auto_approve=self.settings.auto_approve_edits,
                )
            except Exception as exc:
                logger.exception("Agent failed")
                self._set_busy(False, "")
                self._error(str(exc))
                self._focus_composer()
                return
            self._set_busy(False, "")
            self._render_agent_result(card, activity, result)

        threading.Thread(target=work, daemon=True).start()

    def _render_agent_result(self, card, activity, result) -> None:
        """Show the agent's answer, and prompt for approval if it paused."""
        if result.text:
            activity.controls.append(
                ft.Container(
                    padding=ft.padding.only(top=8),
                    content=self._markdown(result.text),
                )
            )

        meta = []
        if result.used_tools:
            meta.append("{} tool calls".format(len(result.used_tools)))
        if result.input_tokens:
            meta.append("${:.3f}".format(result.cost_usd))
        if result.stopped_early:
            # An API failure already explains itself in the answer text; only
            # a genuine step-limit stop needs the extra note.
            errored = any(s.kind == "error" for s in result.steps)
            meta.append("stopped" if errored else "stopped at the step limit")
        if meta:
            activity.controls.append(
                ft.Text(" · ".join(meta), size=10, color=MUTED)
            )

        self._safe_update()

        if result.pending is not None:
            self._ask_approval(result.pending)
        else:
            self._refresh_sidebar()
            self._focus_composer()

    def _ask_approval(self, pending) -> None:
        """Confirm a destructive tool call before it runs."""

        def decide(approved: bool):
            self.page.close(dialog)
            self._set_busy(True, "Working…" if approved else "Cancelling…")

            activity = ft.Column(spacing=3, tight=True)
            card = ft.Container(
                bgcolor=PANEL,
                border_radius=12,
                padding=ft.padding.symmetric(horizontal=16, vertical=12),
                content=activity,
                expand=True,
            )
            self.transcript.controls.append(
                ft.Row([card], alignment=ft.MainAxisAlignment.START)
            )
            self._safe_update()

            def on_step(step):
                if step.kind != "tool":
                    return
                activity.controls.append(
                    ft.Row(
                        spacing=7,
                        controls=[
                            ft.Icon(ft.Icons.CHEVRON_RIGHT, size=13, color=ACCENT),
                            ft.Text(step.label, size=12, color=MUTED, expand=True),
                        ],
                    )
                )
                self._safe_update()

            def work():
                try:
                    result = self.controller.resume_agent(
                        approved,
                        on_step=on_step,
                        auto_approve=self.settings.auto_approve_edits,
                    )
                except Exception as exc:
                    logger.exception("Resume failed")
                    self._set_busy(False, "")
                    self._error(str(exc))
                    return
                self._set_busy(False, "")
                self._render_agent_result(card, activity, result)

            threading.Thread(target=work, daemon=True).start()

        preview = pending.tool_input.get("content")
        body: List[ft.Control] = [
            ft.Text(pending.summary, size=14, color=TEXT),
        ]
        if preview:
            body.append(
                ft.Container(
                    bgcolor=BG,
                    border_radius=8,
                    padding=ft.padding.all(10),
                    content=ft.Text(
                        preview[:700] + ("…" if len(preview) > 700 else ""),
                        size=11,
                        color=MUTED,
                        selectable=True,
                    ),
                )
            )
        body.append(
            ft.Text(
                "This changes a file in your vault and cannot be undone from "
                "inside Jarvis.",
                size=11,
                color=MUTED,
            )
        )

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Allow this change?"),
            content=ft.Container(
                width=480,
                content=ft.Column(body, tight=True, spacing=10, scroll=ft.ScrollMode.AUTO),
            ),
            actions=[
                ft.TextButton("No", on_click=lambda _: decide(False)),
                ft.FilledButton("Allow", on_click=lambda _: decide(True)),
            ],
        )
        self.page.open(dialog)

    def _verify_key_async(self) -> None:
        """Check a newly saved key with one tiny call on the cheapest model.

        Saving a key that turns out to be wrong is silent otherwise: the UI
        says "saved" and the failure surfaces later, attached to whatever the
        user happened to ask next.
        """
        self._set_status("Checking the API key…")

        def work():
            from desktop.agent import describe_api_error

            try:
                client = self.controller.engine._get_client()  # noqa: SLF001
                client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=4,
                    messages=[{"role": "user", "content": "ok"}],
                )
            except Exception as exc:
                message = describe_api_error(exc)
                self._set_status("Key problem — see the message")
                self._system_message("**API key check failed.**\n\n" + message)
                return
            self._set_status("Settings saved — API key works")

        threading.Thread(target=work, daemon=True).start()

    def _retry_in_agent_mode(self) -> None:
        """Turn on Agent mode and re-send the question that was refused."""
        last = next(
            (t.text for t in reversed(self.controller.history) if t.role == "user"),
            "",
        )
        self.settings.agent_mode = True
        self.agent_switch.value = True
        self.settings.save()
        self.agent_label.value = "Agent"
        self.agent_label.color = ACCENT
        self._safe_update()
        if last:
            self._add_user_message(last)
            self._run_agent(last)

    def _toggle_agent_mode(self, e) -> None:
        self.settings.agent_mode = bool(self.agent_switch.value)
        self.settings.save()
        self.agent_label.value = "Agent" if self.settings.agent_mode else "Ask"
        self.agent_label.color = ACCENT if self.settings.agent_mode else MUTED
        self._set_status(
            "Agent mode — I can search, read, and write notes"
            if self.settings.agent_mode
            else "Ask mode — questions only, nothing is modified"
        )
        self._focus_composer()

    def _new_chat(self, _=None) -> None:
        self.controller.clear_history()
        self.transcript.controls.clear()
        self._safe_update()
        self._focus_composer()

    # --- rendering ----------------------------------------------------------

    def _add_user_message(self, text: str) -> None:
        self.transcript.controls.append(
            ft.Row(
                alignment=ft.MainAxisAlignment.END,
                controls=[
                    ft.Container(
                        bgcolor=USER_BUBBLE,
                        border_radius=12,
                        padding=ft.padding.symmetric(horizontal=16, vertical=11),
                        content=ft.Text(
                            text, color=TEXT, size=self.settings.font_size,
                            selectable=True,
                        ),
                    )
                ],
            )
        )
        self._safe_update()

    def _markdown(self, text: str) -> ft.Markdown:
        return ft.Markdown(
            text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            code_theme=ft.MarkdownCodeTheme.ATOM_ONE_DARK,
            on_tap_link=lambda e: self.page.launch_url(e.data),
        )

    def _add_answer(self, answer) -> None:
        refused = not answer.in_scope
        body: List[ft.Control] = [self._markdown(answer.text or "")]

        # A scope refusal in Ask mode looks like a dead end unless the way
        # out is visible. Agent mode has no gate and can search the web, so
        # offer the switch right where the refusal appears rather than
        # leaving the user to find the toggle.
        if answer.decision == "out_of_scope" and not self.settings.agent_mode:
            body.append(
                ft.Row(
                    spacing=8,
                    controls=[
                        ft.Text(
                            "Ask mode only answers from your notes.",
                            size=11,
                            color=MUTED,
                        ),
                        ft.TextButton(
                            "Switch to Agent and retry",
                            style=ft.ButtonStyle(color=ACCENT),
                            on_click=lambda e: self._retry_in_agent_mode(),
                        ),
                    ],
                )
            )

        if answer.citations:
            body.append(
                ft.ExpansionTile(
                    title=ft.Text(
                        "Sources · {}".format(len(answer.citations)),
                        size=12, color=MUTED,
                    ),
                    controls=[
                        ft.Container(
                            padding=ft.padding.only(left=12, right=12, bottom=8),
                            content=ft.Column(
                                spacing=8,
                                controls=[
                                    self._citation_row(i, c)
                                    for i, c in enumerate(answer.citations, 1)
                                ],
                            ),
                        )
                    ],
                )
            )

        # The gate's reasoning is shown rather than hidden: a refusal the user
        # cannot explain reads as the tool being broken.
        meta_bits = []
        if answer.decision:
            meta_bits.append(str(answer.decision).replace("_", " "))
        if answer.best_score:
            meta_bits.append("score {:.2f}".format(answer.best_score))
        if getattr(answer, "documents_in_scope", 0):
            meta_bits.append("{} notes in scope".format(answer.documents_in_scope))
        if meta_bits:
            body.append(ft.Text(" · ".join(meta_bits), size=10, color=MUTED))

        self.transcript.controls.append(
            ft.Row(
                alignment=ft.MainAxisAlignment.START,
                controls=[
                    ft.Container(
                        bgcolor=REFUSAL if refused else PANEL,
                        border_radius=12,
                        padding=ft.padding.symmetric(horizontal=16, vertical=12),
                        content=ft.Column(body, spacing=8, tight=True),
                        expand=True,
                    )
                ],
            )
        )
        self._safe_update()

    def _citation_row(self, n: int, citation) -> ft.Control:
        tags = ", ".join(citation.retrievers) or "—"
        # Citations carry the vault path, so a source can be opened and read
        # in full rather than judged from a 320-character excerpt.
        vault_path = citation.metadata.get("vault_path")
        return ft.Container(
            bgcolor=BG,
            border_radius=8,
            padding=ft.padding.all(10),
            on_click=(
                (lambda e, p=vault_path: self._open_note(p)) if vault_path else None
            ),
            tooltip="Open {}".format(vault_path) if vault_path else None,
            content=ft.Column(
                spacing=4,
                tight=True,
                controls=[
                    ft.Row(
                        [
                            ft.Text(
                                "{}. {}".format(n, citation.doc_name),
                                size=12, weight=ft.FontWeight.W_600, color=TEXT,
                                expand=True,
                            ),
                            ft.Text(tags, size=10, color=MUTED),
                        ]
                    ),
                    ft.Text(citation.excerpt, size=11, color=MUTED, selectable=True),
                ],
            ),
        )

    def _system_message(self, markdown: str) -> None:
        self.transcript.controls.append(
            ft.Container(
                bgcolor=PANEL,
                border_radius=12,
                padding=ft.padding.symmetric(horizontal=18, vertical=14),
                content=self._markdown(markdown),
            )
        )
        self._safe_update()

    def _error(self, message: str) -> None:
        self.page.open(
            ft.SnackBar(
                content=ft.Text(message, color=TEXT),
                bgcolor="#5a2222",
            )
        )
        self._safe_update()

    # --- scrolling ------------------------------------------------------

    def _on_transcript_scroll(self, e) -> None:
        """Show the jump button once the user scrolls away from the bottom."""
        try:
            distance_from_bottom = e.max_scroll_extent - e.pixels
        except Exception:
            return
        should_show = distance_from_bottom > 240
        if should_show != self.jump_button.visible:
            self.jump_button.visible = should_show
            # auto_scroll is suspended while reading back, so a new message
            # cannot yank the view away mid-sentence.
            self.transcript.auto_scroll = (
                self.settings.auto_scroll and not should_show
            )
            self._safe_update()

    def _jump_to_latest(self, _=None) -> None:
        self.jump_button.visible = False
        self.transcript.auto_scroll = self.settings.auto_scroll
        try:
            self.transcript.scroll_to(offset=-1, duration=260)
        except Exception:
            logger.debug("scroll_to unavailable", exc_info=True)
        self._safe_update()

    # --- settings ---------------------------------------------------------

    def _open_settings(self, _=None) -> None:
        key_field = ft.TextField(
            label="Anthropic API key",
            value=load_api_key(),
            password=True,
            can_reveal_password=True,
            hint_text="sk-ant-…",
            helper_text="Stored in the macOS Keychain, not in your vault.",
        )
        floor = ft.Slider(
            min=0.20, max=0.70, divisions=10,
            value=self.settings.relevance_floor,
            label="{value}",
        )
        font = ft.Slider(
            min=12, max=20, divisions=8,
            value=self.settings.font_size,
            label="{value}",
        )
        watch = ft.Switch(
            label="Re-index notes as I edit them",
            value=self.settings.watch_vault,
        )
        auto_approve = ft.Switch(
            label="Let the agent overwrite and delete without asking",
            value=self.settings.auto_approve_edits,
        )
        autoscroll = ft.Switch(
            label="Follow new messages automatically",
            value=self.settings.auto_scroll,
        )
        excluded = ft.TextField(
            label="Folders to skip",
            value=", ".join(self.settings.excluded_folders),
            helper_text="Comma separated. Past AI chat logs are skipped by default.",
        )

        def save(_):
            save_api_key(key_field.value or "")
            self.settings.relevance_floor = round(float(floor.value), 2)
            self.settings.font_size = int(font.value)
            self.settings.watch_vault = bool(watch.value)
            self.settings.auto_approve_edits = bool(auto_approve.value)
            self.settings.auto_scroll = bool(autoscroll.value)
            self.settings.excluded_folders = [
                p.strip() for p in (excluded.value or "").split(",") if p.strip()
            ]
            self.settings.save()

            self.composer.text_size = self.settings.font_size
            self.transcript.auto_scroll = self.settings.auto_scroll
            key_now = load_api_key()
            if self.controller.settings:
                self.controller.settings.relevance_floor = self.settings.relevance_floor
                self.controller.settings.anthropic_api_key = key_now
                # Force a fresh Anthropic client so a newly entered key is
                # used without restarting the app.
                self.controller.engine._client = None  # noqa: SLF001
                # A pending approval belongs to a conversation that failed
                # under the old key; resuming it would replay half a turn.
                agent = self.controller._agent  # noqa: SLF001
                if agent is not None and agent.awaiting_approval:
                    agent.clear()
            self._start_watcher()
            self.page.close(dialog)
            self._focus_composer()

            # Confirm the key actually works rather than letting the user
            # discover a typo on their next message, after the UI has already
            # said "saved". The probe is a few tokens on the cheapest model.
            if key_now:
                self._verify_key_async()
            else:
                self._set_status("Settings saved — no API key set")

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Settings"),
            content=ft.Container(
                width=460,
                content=ft.Column(
                    tight=True,
                    scroll=ft.ScrollMode.AUTO,
                    controls=[
                        key_field,
                        ft.Divider(),
                        ft.Text("Strictness", size=12, color=MUTED),
                        ft.Text(
                            "Higher refuses more questions without spending "
                            "tokens; lower answers more.",
                            size=11, color=MUTED,
                        ),
                        floor,
                        ft.Divider(),
                        ft.Text("Text size", size=12, color=MUTED),
                        font,
                        ft.Divider(),
                        watch,
                        autoscroll,
                        ft.Divider(),
                        ft.Text("Agent", size=12, color=MUTED),
                        ft.Text(
                            "Creating and appending never need approval — they "
                            "cannot destroy anything. Overwriting and deleting "
                            "always ask, unless you turn that off here.",
                            size=11,
                            color=MUTED,
                        ),
                        auto_approve,
                        ft.Divider(),
                        excluded,
                    ],
                ),
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self.page.close(dialog)),
                ft.FilledButton("Save", on_click=save),
            ],
        )
        self.page.open(dialog)

    # --- helpers -----------------------------------------------------------

    def _key_hint(self) -> str:
        if self.controller.has_api_key:
            return ""
        return (
            "\n\n> No API key set yet, so answers cannot be written. "
            "Searching and scoping still work. Add a key in Settings."
        )

    def _on_key(self, e: ft.KeyboardEvent) -> None:
        cmd = e.meta or e.ctrl
        if not cmd:
            if e.key == "Escape" and self.open_note_path:
                self._close_note()
            return

        if e.key == "N":
            self._new_folder() if e.shift else self._new_note()
        elif e.key == "T":
            self._new_chat()
        elif e.key == "S" and self.open_note_path and self.editing:
            self._save_note()
        elif e.key == "E" and self.open_note_path:
            self._toggle_edit()
        elif e.key == "R":
            self._on_rescan()
        elif e.key == ",":
            self._open_settings()
        elif e.key == "K":
            if self.open_note_path:
                self._close_note()
            self._focus_composer()

    def _focus_composer(self) -> None:
        try:
            self.composer.focus()
            self._safe_update()
        except Exception:
            pass

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.progress.visible = busy
        self.send_button.disabled = busy
        self.composer.disabled = busy
        if status:
            self.status_text.value = status
        elif not busy:
            self.status_text.value = ""
        self._safe_update()

    def _set_status(self, text: str) -> None:
        self.status_text.value = text
        self._safe_update()

    def _safe_update(self) -> None:
        """Update from any thread without crashing if the window is closing."""
        try:
            self.page.update()
        except Exception:
            logger.debug("page.update() after teardown", exc_info=True)

    def shutdown(self) -> None:
        if self.watcher is not None:
            self.watcher.stop()
            self.watcher = None


def main(page: ft.Page) -> None:
    app = JarvisApp(page)
    page.on_close = lambda _: app.shutdown()
