"""Chat-style desktop UI (Flet).

Modern generic chat look shared by AI chatbots: one neutral dark-graphite
surface shared by the window and the menu, a white accent, soft rounded
bubbles with an assistant avatar, a pill composer at the bottom and
clickable suggestion chips.

Chat history is real — every conversation is stored in ``chats.json`` and can
be reopened, renamed or deleted from the sidebar. All UI strings are in
English.
"""

from __future__ import annotations

import sys
import threading
import time
import webbrowser
from pathlib import Path

if __package__ in (None, ""):  # also runnable as: python app/main.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flet as ft

from app import chat_store, config
from app.ai_providers import AIProviderError, get_provider
from app.filters import DEFAULT_TIME_RANGE, TIME_RANGES
from app.pipeline import PipelineError, run_pipeline
from app.prompts import LENGTH_PRESETS
from app.window_title import apply_title_bar_theme

WINDOW_TITLE = "NewsProject - AI news digest"

# ------------------------------- palette ---------------------------------- #
BG = "#1c1c1e"          # window + chat background: neutral dark graphite
SIDEBAR_BG = BG         # the menu shares the window colour — one whole
FILL = "#2a2a2e"        # inputs, tonal buttons, chips
SURFACE = "#232326"     # assistant bubble
TEXT = "#ececec"
MUTED = "#9a9aa0"
GRAY = MUTED            # legacy alias used across the file
ACCENT = "#ffffff"      # white accent: primary buttons, tiles, links
ACCENT_SOFT = "#313136"  # active row / avatar background
ACCENT_HOVER = "#3b3b42"  # hover of an active row
USER_BUBBLE_BG = "#34343a"
GREEN = "#3ddb84"       # availability status only (not an accent)
RED = "#ff7575"

SIDEBAR_WIDTH = 300
CONTENT_MAX_WIDTH = 880   # chat column never gets wider than this
RADIUS = 10
FIELD_RADIUS = 12
BTN_STYLE = ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=RADIUS))

STATUS_META = {
    "available": (GREEN, "available"),
    "unavailable": (RED, "unavailable"),
}


def _invisible_outline() -> ft.OutlineInputBorder:
    """A rounded outline whose side is never painted, in every state.

    ``border=None`` still draws the theme default outline (plus a focus
    ring) and ``NoInputBorder()`` paints the fill with square corners, so
    the rounded "pill" input look needs an OutlineInputBorder with an
    explicitly unpainted side — for DEFAULT *and* FOCUSED, otherwise the
    theme would resolve the focused border by itself.
    """
    return ft.OutlineInputBorder(
        border_radius=FIELD_RADIUS,
        side=ft.BorderSide(style=ft.BorderStyle.NONE),
    )


# Shared input style: soft rounded fill, no visible border anywhere.
_BORDERLESS = {
    "border": {
        ft.ControlState.DEFAULT: _invisible_outline(),
        ft.ControlState.FOCUSED: _invisible_outline(),
        ft.ControlState.ERROR: _invisible_outline(),
        ft.ControlState.DISABLED: _invisible_outline(),
    },
    "filled": True,
    "fill_color": FILL,
}

# Starter prompts on the empty screen: (label, topic or "" for highlights).
STARTERS = [
    ("Top highlights", ""),
    ("AI & tech", "AI, искусственный интеллект"),
    ("Economy", "экономика"),
    ("World", "мир"),
    ("Sport", "спорт"),
]

_MD_STYLE = ft.MarkdownStyleSheet(
    p_text_style=ft.TextStyle(size=14, height=1.45, color=TEXT),
    a_text_style=ft.TextStyle(size=14, color=ACCENT),
    h1_text_style=ft.TextStyle(size=20, weight=ft.FontWeight.W_700, color=TEXT),
    h2_text_style=ft.TextStyle(size=17, weight=ft.FontWeight.W_700, color=TEXT),
    h3_text_style=ft.TextStyle(size=15, weight=ft.FontWeight.W_600, color=TEXT),
    strong_text_style=ft.TextStyle(weight=ft.FontWeight.W_600, color=TEXT),
    list_bullet_text_style=ft.TextStyle(size=14, color=TEXT),
    code_text_style=ft.TextStyle(size=13, color="#cdd0d5"),
    blockquote_decoration=ft.BoxDecoration(bgcolor=FILL),
    blockquote_padding=ft.Padding.symmetric(horizontal=12, vertical=8),
    codeblock_decoration=ft.BoxDecoration(bgcolor=FILL),
    codeblock_padding=ft.Padding.symmetric(horizontal=12, vertical=8),
    block_spacing=6,
)


def _field(**kwargs) -> ft.TextField:
    return ft.TextField(**{**_BORDERLESS, **kwargs})


def _section(label: str) -> ft.Container:
    """Group header: more air above it than below, so it hugs its controls."""
    return ft.Container(
        content=ft.Text(label, size=11, weight=ft.FontWeight.W_700, color=MUTED),
        padding=ft.Padding.only(top=10, bottom=0),
    )


def _gap(height: int) -> ft.Container:
    return ft.Container(height=height)


def _hovered(event) -> bool:
    """True when a pointer entered the control (flet 1.x passes a bool)."""
    data = getattr(event, "data", None)
    if isinstance(data, bool):
        return data
    return str(data).lower() in ("true", "1", "hover")


# First word of a text that should be *answered* rather than used as a filter.
_QUESTION_FIRST = {
    "что", "чего", "как", "почему", "зачем", "кто", "где", "когда", "какой",
    "какая", "какие", "сколько", "будет", "стоит", "можно", "правда",
    "насколько", "чей", "согласны", "почему",
    "what", "how", "why", "when", "who", "where", "which", "should", "will",
    "is", "are", "did", "does", "do", "can",
}


def _looks_like_question(text: str) -> bool:
    """True when the composer text is a question, not a tag list.

    Tag lists filter the articles; a real question is passed to the model
    verbatim — filtering articles against a whole sentence would match
    nothing, and the user would never get an answer.
    """
    t = (text or "").strip()
    if not t:
        return False
    if "?" in t:
        return True
    return t.split()[0].lower().strip(",.!:;") in _QUESTION_FIRST


def main(page: ft.Page) -> None:
    settings = config.load_settings()
    profiles: list[dict] = list(settings.get("api_profiles") or [])
    active_id: str = settings.get("active_api_id", "")
    chats: list[dict] = chat_store.load_chats()
    state = {
        "running": False,
        "dialog_open": False,
        "editing_id": None,   # profile id in the API dialog, None = adding
        "rename_id": None,    # chat id in the rename dialog
        "dialog_test_status": "unknown",
        "testing": False,
        "pending": None,      # parts of the assistant bubble being generated
        "pairs": 0,           # Q/A pairs currently rendered in the thread
        "current_id": None,   # chat opened right now
    }
    if profiles and not any(p.get("id") == active_id for p in profiles):
        active_id = profiles[0].get("id", "")

    page.title = WINDOW_TITLE
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 1280
    page.window.height = 860
    page.window.min_width = 960
    page.window.min_height = 600
    page.window.bgcolor = BG
    page.bgcolor = BG
    page.padding = 0
    page.spacing = 0
    apply_title_bar_theme(WINDOW_TITLE, BG, TEXT, FILL)

    def _safe_update() -> None:
        try:
            page.update()
        except Exception:  # the window may have been closed mid-run
            pass

    def _run(fn) -> None:
        try:
            page.run_thread(fn)
        except Exception:
            threading.Thread(target=fn, daemon=True).start()

    def _reveal_on_hover(
        item: ft.Container,
        base_color: str | None,
        hover_color: str | None,
        reveal: list[ft.Control],
        idle_opacity: float = 0.0,
    ):
        """Highlight a list row on hover and fade its action buttons in."""
        def handler(event) -> None:
            active = _hovered(event)
            item.bgcolor = hover_color if active else base_color
            for control in reveal:
                control.opacity = 1.0 if active else idle_opacity
            _safe_update()

        return handler

    # --------------------------- sidebar inputs --------------------------- #
    _time_range = settings.get("time_range") or DEFAULT_TIME_RANGE
    if _time_range not in TIME_RANGES:
        _time_range = DEFAULT_TIME_RANGE

    sources_field = _field(
        value=settings["sources"],
        hint_text="https://lenta.ru\n@channel\nhttps://t.me/s/channel",
        multiline=True,
        min_lines=3,
        max_lines=7,
        text_size=13,
    )
    time_dd = ft.Dropdown(
        value=_time_range,
        options=[
            ft.DropdownOption(key=k, text=label)
            for k, (label, _delta) in TIME_RANGES.items()
        ],
        text_size=13,
        **_BORDERLESS,
    )
    length_dd = ft.Dropdown(
        value=settings.get("gen_length") or "medium",
        options=[
            ft.DropdownOption(key=k, text=label)
            for k, (label, _chars) in LENGTH_PRESETS.items()
        ],
        text_size=13,
        **_BORDERLESS,
    )
    style_field = _field(
        value=str(settings.get("gen_style") or ""),
        hint_text="e.g. in bullet points, focus on economics",
        text_size=13,
    )

    # ------------------------------ API list ------------------------------ #
    api_list = ft.Column(spacing=4)

    def _active_profile() -> dict | None:
        return next((p for p in profiles if p.get("id") == active_id), None)

    def _render_api_list() -> None:
        api_list.controls.clear()
        for p in profiles:
            color, label = STATUS_META.get(
                p.get("status", "unknown"), (GRAY, "not checked")
            )
            is_active = p.get("id") == active_id
            edit_btn = ft.IconButton(
                ft.Icons.EDIT_OUTLINED,
                icon_size=14,
                tooltip="Edit",
                opacity=0.0,
                style=BTN_STYLE,
                on_click=lambda e, pid=p["id"]: open_api_dialog(pid),
            )
            del_btn = ft.IconButton(
                ft.Icons.DELETE_OUTLINE,
                icon_size=14,
                tooltip="Delete",
                opacity=0.0,
                style=BTN_STYLE,
                on_click=lambda e, pid=p["id"]: delete_profile(pid),
            )
            # Name + status share one text line: two separate Texts with
            # different font sizes can never sit on the same baseline.
            row = ft.Row(
                [
                    ft.Container(width=8, height=8, bgcolor=color, border_radius=4),
                    ft.Text(
                        spans=[
                            ft.TextSpan(
                                p.get("name") or "API",
                                style=ft.TextStyle(size=13, color=TEXT),
                                opacity=1.0 if is_active else 0.8,
                            ),
                            ft.TextSpan("   "),
                            ft.TextSpan(
                                label,
                                style=ft.TextStyle(size=10, color=color),
                            ),
                        ],
                        size=13,
                        max_lines=1,
                        no_wrap=True,
                        expand=True,
                    ),
                    edit_btn,
                    del_btn,
                ],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
            item = ft.Container(
                content=row,
                bgcolor=ACCENT_SOFT if is_active else None,
                border_radius=RADIUS,
                padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                on_click=lambda e, pid=p["id"]: select_profile(pid),
            )
            item.on_hover = _reveal_on_hover(
                item,
                ACCENT_SOFT if is_active else None,
                ACCENT_HOVER if is_active else FILL,
                [edit_btn, del_btn],
            )
            api_list.controls.append(item)
        if not profiles:
            api_list.controls.append(
                ft.Text("No API added yet.", size=12, color=GRAY)
            )

        _safe_update()

    add_api_btn = ft.FilledButton(
        "Add API",
        icon=ft.Icons.ADD,
        height=36,
        expand=True,
        bgcolor=ACCENT,
        color=BG,
        icon_color=BG,
        style=BTN_STYLE,
        on_click=lambda e: open_api_dialog(None),
    )
    refresh_status_btn = ft.IconButton(
        ft.Icons.REFRESH,
        icon_size=15,
        tooltip="Re-check availability of all APIs",
        style=BTN_STYLE,
        on_click=lambda e: check_all_profiles(),
    )

    # ------------------------------ history ------------------------------- #
    history_list = ft.Column(spacing=2)
    new_chat_btn = ft.FilledTonalButton(
        "New chat",
        icon=ft.Icons.ADD,
        height=36,
        expand=True,
        bgcolor=FILL,
        color=TEXT,
        icon_color=ACCENT,
        style=BTN_STYLE,
        on_click=lambda e: new_chat(),
    )

    # ------------------------------- sidebar ------------------------------ #
    # The only place to toggle the menu from: there is no top bar anymore.
    burger_btn = ft.IconButton(
        ft.Icons.MENU,
        tooltip="Toggle sidebar",
        icon_size=20,
        style=BTN_STYLE,
        on_click=lambda e: toggle_sidebar(),
    )
    sidebar = ft.Container(
        width=SIDEBAR_WIDTH,
        bgcolor=SIDEBAR_BG,
        padding=ft.Padding.only(left=12, top=12, right=12, bottom=12),
        content=ft.Column(
            spacing=6,
            scroll="auto",
            expand=True,
            controls=[
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Icon(
                                ft.Icons.NEWSPAPER, size=15, color=BG
                            ),
                            width=26,
                            height=26,
                            border_radius=8,
                            bgcolor=ACCENT,
                            alignment=ft.Alignment.CENTER,
                        ),
                        ft.Text(
                            "NewsProject",
                            size=14,
                            weight=ft.FontWeight.W_700,
                            color=TEXT,
                        ),
                        ft.Container(expand=True),
                        burger_btn,
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                new_chat_btn,
                _section("SOURCES"),
                sources_field,
                _section("TIME RANGE"),
                time_dd,
                _section("SUMMARY"),
                length_dd,
                style_field,
                _section("API"),
                api_list,
                ft.Row(
                    [add_api_btn, refresh_status_btn],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                _section("HISTORY"),
                history_list,
            ],
        ),
    )

    # ------------------------------- top bar ------------------------------ #
    # Removed on purpose: the burger lives in the menu header and a second
    # copy appears above the chat only while the menu is hidden.

    # -------------------------------- chat -------------------------------- #
    def _open_link(event) -> None:
        url = str(getattr(event, "data", "") or "")
        if url.startswith(("http://", "https://")):
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - a missing browser is not fatal
                pass

    def _markdown(value: str, visible: bool) -> ft.Markdown:
        return ft.Markdown(
            value or "",
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            selectable=True,
            md_style_sheet=_MD_STYLE,
            on_tap_link=_open_link,
            visible=visible,
        )

    def _avatar() -> ft.Container:
        return ft.Container(
            content=ft.Icon(ft.Icons.AUTO_AWESOME, size=15, color=ACCENT),
            width=28,
            height=28,
            border_radius=14,
            bgcolor=ACCENT_SOFT,
            alignment=ft.Alignment.CENTER,
            tooltip="NewsProject",
        )

    def _suggestions_block(topics: list[str]) -> ft.Column:
        chips = [
            ft.Chip(
                label=topic,
                bgcolor=FILL,
                label_text_style=ft.TextStyle(size=13, color=TEXT),
                shape=ft.RoundedRectangleBorder(radius=14),
                tooltip=f"Search news about “{topic}”",
                on_click=lambda e, t=topic: quick_ask(t),
            )
            for topic in topics[:10]
        ]
        return ft.Column(
            controls=[
                ft.Text(
                    "Nothing on this topic — pick one of the topics I did find:",
                    size=12,
                    color=GRAY,
                ),
                ft.Row(chips, wrap=True, spacing=6, run_spacing=6),
            ],
            spacing=6,
        )

    def _assistant_row(
        md_text: str = "",
        stats_text: str = "",
        suggestions: list[str] | None = None,
        thinking: bool = False,
    ) -> dict:
        """One assistant message (live or restored from the history)."""
        ring = status = progress = thinking_row = None
        if thinking:
            ring = ft.ProgressRing(
                width=18, height=18, stroke_width=2, color=ACCENT
            )
            status = ft.Text("Starting...", size=12, color=GRAY)
            progress = ft.ProgressBar(
                value=0.0, width=170, bar_height=3, color=ACCENT, bgcolor=FILL
            )
            thinking_row = ft.Row(
                [
                    ring,
                    ft.Column([status, progress], spacing=6),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

        md = _markdown(md_text, visible=bool(md_text) and not thinking)
        stats_control = ft.Text(
            stats_text, size=11, color=GRAY, visible=bool(stats_text) and not thinking
        )
        chips_holder = ft.Column(controls=[], spacing=6, visible=False)
        if suggestions:
            chips_holder.controls.append(_suggestions_block(list(suggestions)))
            chips_holder.visible = True

        controls = [
            c for c in (thinking_row, md, stats_control, chips_holder) if c is not None
        ]
        bubble = ft.Container(
            # expand + expand_loose: the bubble takes at most the space left
            # by the avatar, so long Markdown wraps instead of overflowing.
            expand=True,
            expand_loose=True,
            content=ft.Column(controls=controls, spacing=10),
            bgcolor=SURFACE,
            border_radius=ft.BorderRadius.only(
                top_left=4, top_right=16, bottom_left=16, bottom_right=16
            ),
            padding=ft.Padding.symmetric(horizontal=16, vertical=12),
        )
        row = ft.Row(
            [_avatar(), bubble],
            spacing=10,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
            alignment=ft.MainAxisAlignment.START,
        )
        return {
            "row": row,
            "md": md,
            "stats": stats_control,
            "chips": chips_holder,
            "thinking": thinking_row,
            "ring": ring,
            "status": status,
            "progress": progress,
        }

    def _user_row(message: dict) -> ft.Row:
        label = message.get("label") or message.get("text") or "Top highlights"
        if label.startswith("Topic: "):  # legacy prefix from older chats.json
            label = label[len("Topic: "):]
        return ft.Row(
            [
                ft.Container(
                    # loose expansion keeps the bubble right-aligned while
                    # capping its width so long topics wrap instead of
                    # pushing the row wider than the thread.
                    expand=True,
                    expand_loose=True,
                    content=ft.Text(label, size=14, selectable=True),
                    bgcolor=USER_BUBBLE_BG,
                    padding=ft.Padding.symmetric(horizontal=16, vertical=10),
                    border_radius=ft.BorderRadius.only(
                        top_left=16, top_right=16, bottom_right=6, bottom_left=16
                    ),
                )
            ],
            alignment=ft.MainAxisAlignment.END,
            tight=True,
        )

    def _message_args(message: dict) -> dict:
        return {
            "md_text": str(message.get("text") or ""),
            "stats_text": str(message.get("stats") or ""),
            "suggestions": list(message.get("suggestions") or []),
            "thinking": False,
        }

    def _empty_state() -> ft.Container:
        chips = [
            ft.Chip(
                label=label,
                bgcolor=FILL,
                label_text_style=ft.TextStyle(size=13, color=TEXT),
                shape=ft.RoundedRectangleBorder(radius=14),
                tooltip="Ask for this",
                on_click=lambda e, t=topic: quick_ask(t),
            )
            for label, topic in STARTERS
        ]
        return ft.Container(
            expand=True,
            alignment=ft.Alignment.CENTER,
            padding=ft.Padding.symmetric(horizontal=24, vertical=40),
            content=ft.Column(
                spacing=8,
                expand=True,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Container(
                        content=ft.Icon(
                            ft.Icons.AUTO_AWESOME, size=26, color=BG
                        ),
                        width=52,
                        height=52,
                        border_radius=16,
                        bgcolor=ACCENT,
                        alignment=ft.Alignment.CENTER,
                    ),
                    ft.Text(
                        "What's in the news?",
                        size=24,
                        weight=ft.FontWeight.W_700,
                        color=TEXT,
                    ),
                    ft.Text(
                        "Ask a topic — or press Enter for the top highlights "
                        "from your sources.",
                        size=13,
                        color=MUTED,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    _gap(6),
                    ft.Row(
                        chips,
                        wrap=True,
                        spacing=6,
                        run_spacing=6,
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                ],
            ),
        )

    thread = ft.Column(
        expand=True,
        scroll="auto",
        spacing=16,
        auto_scroll=True,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        controls=[_empty_state()],
    )

    topic_field = _field(
        hint_text="Ask about news — topic or tags (optional)",
        expand=True,
        text_size=14,
        min_lines=1,
        max_lines=5,
        multiline=True,
        # The field box is smaller than the composer pill it sits in, so the
        # theme's hover/focus fill would light up only a part of the pill —
        # keep both states clear and let the pill itself do the highlighting.
        fill_color=ft.Colors.TRANSPARENT,
        hover_color=ft.Colors.TRANSPARENT,
        focus_color=ft.Colors.TRANSPARENT,
        content_padding=ft.Padding.symmetric(horizontal=10, vertical=12),
        on_submit=lambda e: send(),
    )
    def _send_hover(event) -> None:
        send_btn.bgcolor = "#e0e0e4" if _hovered(event) else ACCENT
        _safe_update()

    send_btn = ft.Container(
        content=ft.Icon(ft.Icons.ARROW_UPWARD, size=18, color=BG),
        width=40,
        height=40,
        border_radius=20,  # width == height -> a true circle
        bgcolor=ACCENT,
        # A Material button keeps its 64 px minimum width inside a 40 px box,
        # which pushes the arrow to the right; a container centres it exactly.
        alignment=ft.Alignment.CENTER,
        tooltip="Send",
        on_click=lambda e: send(),
    )
    send_btn.on_hover = _send_hover
    composer = ft.Container(
        content=ft.Row(
            [topic_field, send_btn],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        bgcolor=FILL,
        border_radius=24,
        padding=ft.Padding.only(left=6, top=4, right=6, bottom=4),
    )
    composer_wrap = ft.Container(
        content=composer,
        padding=ft.Padding.only(bottom=16, top=4),
    )
    # Shown only while the menu is hidden — the only way to bring it back
    # once the top bar is gone.
    chat_burger = ft.IconButton(
        ft.Icons.MENU,
        tooltip="Toggle sidebar",
        icon_size=20,
        style=BTN_STYLE,
        on_click=lambda e: toggle_sidebar(),
    )
    chat_head = ft.Row(
        [chat_burger],
        alignment=ft.MainAxisAlignment.START,
        visible=False,
    )
    chat_col = ft.Column(
        # the chat must not start flush with the top of the window
        controls=[
            ft.Container(height=20),
            chat_head,
            ft.Container(height=8),
            thread,
            composer_wrap,
        ],
        expand=True,
        spacing=0,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # ----------------------------- API dialog ----------------------------- #
    # "Profile name" is the label shown in the sidebar list; "Model" is the
    # model id actually sent to the provider — they are not the same thing.
    api_name_field = _field(
        label="Profile name", hint_text="e.g. Work account", text_size=13
    )
    dlg_base_url = _field(
        label="Base URL", hint_text="https://api.giga.chat/v1", text_size=13
    )
    dlg_api_key = _field(
        label="API key (optional for local models)", password=True, text_size=13
    )
    dlg_model = _field(label="Model", hint_text="GigaChat-2", text_size=13)

    test_result = ft.Text("", size=12, color=GRAY)
    test_btn = ft.FilledTonalButton(
        "Test connection",
        height=36,
        bgcolor=FILL,
        color=TEXT,
        style=BTN_STYLE,
        on_click=lambda e: start_test(),
    )
    delete_btn = ft.TextButton(
        "Delete",
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=RADIUS), color=RED),
        on_click=lambda e: delete_from_dialog(),
        visible=False,
    )

    api_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Add API", size=15, weight=ft.FontWeight.W_600),
        bgcolor=SURFACE,
        content=ft.Column(
            spacing=10,
            scroll="auto",
            width=420,
            # a scrolling column otherwise expands to the full window height
            # and drags the whole dialog along with it
            height=272,
            controls=[
                api_name_field,
                dlg_base_url,
                dlg_api_key,
                dlg_model,
                ft.Row(
                    [test_btn, test_result],
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
        ),
        actions=[
            delete_btn,
            ft.FilledTonalButton(
                "Cancel",
                height=36,
                bgcolor=FILL,
                color=TEXT,
                style=BTN_STYLE,
                on_click=lambda e: close_dialog(),
            ),
            ft.FilledButton(
                "Save",
                height=36,
                bgcolor=ACCENT,
                color=BG,
                style=BTN_STYLE,
                on_click=lambda e: save_api(),
            ),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    # --------------------------- rename dialog ---------------------------- #
    rename_field = _field(label="Chat title", text_size=13, max_length=80)
    rename_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("Rename chat", size=15, weight=ft.FontWeight.W_600),
        bgcolor=SURFACE,
        content=ft.Column(spacing=10, width=380, controls=[rename_field]),
        actions=[
            ft.FilledTonalButton(
                "Cancel",
                height=36,
                bgcolor=FILL,
                color=TEXT,
                style=BTN_STYLE,
                on_click=lambda e: close_dialog(),
            ),
            ft.FilledButton(
                "Save",
                height=36,
                bgcolor=ACCENT,
                color=BG,
                style=BTN_STYLE,
                on_click=lambda e: save_rename(),
            ),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    # ------------------------------ handlers ------------------------------ #
    def on_status(text: str) -> None:
        pending = state.get("pending")
        status = pending.get("status") if pending else None
        if status is not None:
            status.value = text
            _safe_update()

    def on_progress(value: float) -> None:
        pending = state.get("pending")
        progress = pending.get("progress") if pending else None
        if progress is not None:
            progress.value = value
            _safe_update()

    def _persist() -> None:
        snapshot = dict(settings)
        snapshot["sources"] = sources_field.value or ""
        snapshot["time_range"] = time_dd.value or DEFAULT_TIME_RANGE
        snapshot["gen_temperature"] = state.get("gen_temperature", 0.4)
        snapshot["gen_length"] = length_dd.value or "medium"
        snapshot["gen_style"] = (style_field.value or "").strip()
        snapshot["api_profiles"] = profiles
        snapshot["active_api_id"] = active_id
        try:
            config.save_settings(snapshot)
        except OSError:
            pass

    def _persist_chats() -> None:
        try:
            chat_store.save_chats(chats)
        except OSError:
            pass
        _render_history()

    # ------------------------------- API CRUD ----------------------------- #
    def select_profile(pid: str) -> None:
        nonlocal active_id
        if pid != active_id:
            active_id = pid
            _persist()
            _render_api_list()

    def _dialog_profile(pid: str | None) -> dict:
        """Profile dict as currently filled in the dialog (not yet saved)."""
        return {
            "id": pid or config.new_profile_id(),
            "name": api_name_field.value or "API",
            "base_url": dlg_base_url.value or "",
            "api_key": dlg_api_key.value or "",
            "model": dlg_model.value or "",
            "status": state["dialog_test_status"],
            "status_detail": "",
        }

    def check_profile(pid: str) -> None:
        """Background availability check for one saved profile."""
        profile = next((p for p in profiles if p.get("id") == pid), None)
        if profile is None:
            return

        def worker() -> None:
            try:
                _, elapsed = get_provider(
                    config.profile_to_settings(profile)
                ).test_connection()
                profile["status"] = "available"
                profile["status_detail"] = f"{elapsed} ms"
            except Exception as exc:  # noqa: BLE001 - any failure = unavailable
                profile["status"] = "unavailable"
                profile["status_detail"] = str(exc)[:200]
            _persist()
            _render_api_list()

        _run(worker)

    def check_all_profiles(_e=None) -> None:
        for p in list(profiles):
            check_profile(p["id"])

    def delete_profile(pid: str) -> None:
        nonlocal active_id
        profiles[:] = [p for p in profiles if p.get("id") != pid]
        if active_id == pid:
            active_id = profiles[0]["id"] if profiles else ""
        _persist()
        _render_api_list()

    # ------------------------------ API dialog ---------------------------- #
    def open_api_dialog(pid: str | None) -> None:
        if state["dialog_open"]:
            return
        state["editing_id"] = pid
        state["dialog_test_status"] = "unknown"
        profile = next((p for p in profiles if p.get("id") == pid), None)
        if profile:  # edit existing
            api_dialog.title.value = "Edit API"
            api_name_field.value = profile.get("name", "")
            dlg_base_url.value = profile.get("base_url", "")
            dlg_api_key.value = profile.get("api_key", "")
            dlg_model.value = profile.get("model", "")
            delete_btn.visible = True
            state["dialog_test_status"] = profile.get("status", "unknown")
        else:  # add new
            api_dialog.title.value = "Add API"
            api_name_field.value = ""
            dlg_base_url.value = ""
            dlg_api_key.value = ""
            dlg_model.value = ""
            delete_btn.visible = False
        test_result.value = ""
        test_result.color = GRAY
        state["dialog_open"] = True
        page.show_dialog(api_dialog)

    def close_dialog(_e=None) -> None:
        if not state["dialog_open"]:
            return
        state["dialog_open"] = False
        try:
            page.pop_dialog()
        except Exception:
            pass
        _safe_update()

    def save_api(_e=None) -> None:
        nonlocal active_id
        profile = _dialog_profile(state["editing_id"])
        existing = next(
            (i for i, p in enumerate(profiles) if p.get("id") == profile["id"]), None
        )
        if existing is not None:
            profiles[existing] = profile
        else:
            profiles.append(profile)
        if not active_id:
            active_id = profile["id"]
        close_dialog()
        _persist()
        _render_api_list()
        check_profile(profile["id"])  # refresh availability in background

    def delete_from_dialog(_e=None) -> None:
        pid = state["editing_id"]
        close_dialog()
        if pid:
            delete_profile(pid)

    def start_test(_e=None) -> None:
        if state.get("testing"):
            return
        state["testing"] = True
        test_btn.disabled = True
        test_result.value = "Testing..."
        test_result.color = GRAY
        _safe_update()
        candidate = _dialog_profile(state["editing_id"])

        def worker() -> None:
            try:
                reply, elapsed = get_provider(
                    config.profile_to_settings(candidate)
                ).test_connection()
                state["dialog_test_status"] = "available"
                test_result.value = f"OK ({elapsed} ms): {reply[:50]}"
                test_result.color = GREEN
            except AIProviderError as exc:
                state["dialog_test_status"] = "unavailable"
                test_result.value = str(exc)[:120]
                test_result.color = RED
            except Exception as exc:  # noqa: BLE001
                state["dialog_test_status"] = "unavailable"
                test_result.value = f"{type(exc).__name__}: {exc}"[:120]
                test_result.color = RED
            finally:
                state["testing"] = False
                test_btn.disabled = False
                _safe_update()

        _run(worker)

    # ----------------------------- chat history --------------------------- #
    def _current_chat() -> dict | None:
        return next((c for c in chats if c["id"] == state["current_id"]), None)

    def _find_chat(chat_id: str | None) -> dict | None:
        return next((c for c in chats if c["id"] == chat_id), None)

    def _render_history() -> None:
        history_list.controls.clear()
        if not chats:
            history_list.controls.append(
                ft.Text("No chats yet.", size=12, color=GRAY)
            )
            _safe_update()
            return
        for chat in chats[:40]:
            current = chat["id"] == state["current_id"]
            title = ft.Text(
                chat.get("title") or "Chat",
                size=13,
                max_lines=1,
                no_wrap=True,
                color=TEXT,
            )
            stamp = ft.Text(
                chat_store.relative_stamp(chat.get("updated", 0)),
                size=10,
                color=GRAY,
            )
            edit_btn = ft.IconButton(
                ft.Icons.EDIT_OUTLINED,
                icon_size=14,
                tooltip="Rename chat",
                opacity=0.0,
                style=BTN_STYLE,
                on_click=lambda e, cid=chat["id"]: open_rename(cid),
            )
            del_btn = ft.IconButton(
                ft.Icons.DELETE_OUTLINE,
                icon_size=14,
                tooltip="Delete chat",
                opacity=0.0,
                style=BTN_STYLE,
                on_click=lambda e, cid=chat["id"]: delete_chat(cid),
            )
            row = ft.Row(
                [
                    ft.Column(
                        [title, stamp],
                        spacing=1,
                        expand=True,
                    ),
                    edit_btn,
                    del_btn,
                ],
                spacing=2,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
            base = ACCENT_SOFT if current else None
            item = ft.Container(
                content=row,
                bgcolor=base,
                border_radius=RADIUS,
                padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                on_click=lambda e, cid=chat["id"]: open_chat(cid),
            )
            item.on_hover = _reveal_on_hover(
                item,
                base,
                ACCENT_HOVER if current else FILL,
                [edit_btn, del_btn],
            )
            history_list.controls.append(item)
        _safe_update()

    def open_chat(chat_id: str, _e=None) -> None:
        if state["running"]:
            return
        chat = _find_chat(chat_id)
        if chat is None:
            return
        state["current_id"] = chat_id
        _render_thread(chat)
        _render_history()
        _safe_update()

    def new_chat(_e=None) -> None:
        if state["running"]:
            return
        state["current_id"] = None
        state["pending"] = None
        topic_field.value = ""
        _render_thread(None)
        _render_history()
        _safe_update()

    def delete_chat(chat_id: str, _e=None) -> None:
        if state["running"]:
            return
        is_current = state["current_id"] == chat_id
        chats[:] = [c for c in chats if c["id"] != chat_id]
        try:
            chat_store.save_chats(chats)
        except OSError:
            pass
        if is_current:
            state["current_id"] = None
            _render_thread(None)
        _render_history()
        _safe_update()

    def open_rename(chat_id: str, _e=None) -> None:
        if state["dialog_open"] or state["running"]:
            return
        chat = _find_chat(chat_id)
        if chat is None:
            return
        state["rename_id"] = chat_id
        state["dialog_open"] = True
        rename_field.value = chat.get("title", "")
        page.show_dialog(rename_dialog)

    def save_rename(_e=None) -> None:
        chat = _find_chat(state["rename_id"])
        title = (rename_field.value or "").strip()
        if chat and title:
            chat["title"] = title[:80]
            chat["updated"] = time.time()
        close_dialog()
        try:
            chat_store.save_chats(chats)
        except OSError:
            pass
        _render_history()
        _safe_update()

    # ------------------------------ chat thread --------------------------- #
    def _render_thread(chat: dict | None) -> None:
        """Rebuild the visible thread from a stored conversation."""
        thread.controls.clear()
        state["pairs"] = 0
        state["pending"] = None
        messages = (chat or {}).get("messages") or []
        if not messages:
            thread.controls.append(_empty_state())
            return

        index = 0
        while index < len(messages):
            message = messages[index]
            if message.get("role") == chat_store.ROLE_USER:
                partner = None
                if (
                    index + 1 < len(messages)
                    and messages[index + 1].get("role") == chat_store.ROLE_ASSISTANT
                ):
                    partner = messages[index + 1]
                controls = [_user_row(message)]
                if partner is not None:
                    controls.append(_assistant_row(**_message_args(partner))["row"])
                thread.controls.append(
                    ft.Column(
                        controls=controls,
                        spacing=12,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                        key=f"pair-{state['pairs']}",
                    )
                )
                state["pairs"] += 1
                index += 2 if partner is not None else 1
            else:
                parts = _assistant_row(**_message_args(message))
                thread.controls.append(
                    ft.Column(
                        controls=[parts["row"]],
                        spacing=12,
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                        key=f"pair-{state['pairs']}",
                    )
                )
                state["pairs"] += 1
                index += 1

    def _finalize(
        parts: dict,
        markdown: str,
        stats_text: str = "",
        suggestions: list[str] | None = None,
    ) -> None:
        """Replace the thinking placeholder with the final answer."""
        if parts.get("thinking") is not None:
            parts["ring"].visible = False
            parts["thinking"].visible = False
        parts["md"].value = markdown
        parts["md"].visible = True
        if stats_text:
            parts["stats"].value = stats_text
            parts["stats"].visible = True
        if suggestions:
            parts["chips"].controls.clear()
            parts["chips"].controls.append(_suggestions_block(list(suggestions)))
            parts["chips"].visible = True

    def _finish(
        chat_id: str,
        parts: dict,
        markdown: str,
        stats_text: str = "",
        suggestions: list[str] | None = None,
    ) -> None:
        """Show the answer and store it in the conversation."""
        _finalize(parts, markdown, stats_text, suggestions)
        chat = _find_chat(chat_id)
        if chat is not None:
            chat_store.append_message(
                chat,
                {
                    "role": chat_store.ROLE_ASSISTANT,
                    "text": markdown,
                    "stats": stats_text,
                    "suggestions": list(suggestions or []),
                },
            )
            _persist_chats()
        state["running"] = False
        state["pending"] = None
        send_btn.disabled = False
        send_btn.opacity = 1.0
        new_chat_btn.disabled = False
        _safe_update()

    def worker(cfg: dict, parts: dict, chat_id: str) -> None:
        try:
            result = run_pipeline(cfg, on_status, on_progress)
            extras = result.stats
            if result.warnings:
                extras += "  |  " + "; ".join(result.warnings)[:180]
            _finish(chat_id, parts, result.summary, extras, result.suggestions)
        except PipelineError as exc:
            _finish(chat_id, parts, f"**Error:** {exc}")
        except Exception as exc:  # noqa: BLE001 - any failure goes into the chat
            _finish(chat_id, parts, f"**Error:** {type(exc).__name__}: {exc}")

    def quick_ask(topic: str) -> None:
        """Fill the composer with a suggestion and send it right away."""
        if state["running"]:
            return
        topic_field.value = topic
        send()

    def send(_e=None) -> None:
        if state["running"]:
            return
        active = _active_profile()
        raw = (topic_field.value or "").strip()
        label = raw or "Top highlights"

        chat = _current_chat()
        if chat is None:
            chat = chat_store.create_chat(chat_store.title_from(raw))
            chats.insert(0, chat)
            state["current_id"] = chat["id"]

        if state["pairs"] == 0:
            thread.controls.clear()

        chat_store.append_message(
            chat,
            {"role": chat_store.ROLE_USER, "text": raw, "label": label},
        )
        parts = _assistant_row(thinking=True)
        pair = ft.Column(
            controls=[_user_row({"label": label}), parts["row"]],
            spacing=12,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            key=f"pair-{state['pairs']}",
        )
        thread.controls.append(pair)
        state["pairs"] += 1
        topic_field.value = ""

        state["running"] = True
        state["pending"] = parts
        send_btn.disabled = True
        send_btn.opacity = 0.45
        new_chat_btn.disabled = True
        _persist()
        _persist_chats()
        _safe_update()

        if active is None:
            _finish(
                chat["id"],
                parts,
                "**No API selected.** Add one with **+ Add API** in the "
                "sidebar and press Send again.",
            )
            return

        # A question must not become a filter tag: the whole sentence would
        # have to match an article word-for-word. Pass it to the model and
        # let it answer from everything inside the time range instead.
        ask = raw if _looks_like_question(raw) else ""
        cfg = {
            "sources": sources_field.value or "",
            "time_range": time_dd.value or DEFAULT_TIME_RANGE,
            "tags": "" if ask else raw,
            "question": ask,
        }
        cfg.update(config.profile_to_settings(active))
        _run(lambda: worker(cfg, parts, chat["id"]))

    # -------------------------------- layout ------------------------------ #
    def toggle_sidebar(_e=None) -> None:
        sidebar.visible = not sidebar.visible
        chat_head.visible = not sidebar.visible
        _fit_widths()

    def _fit_widths(_e=None) -> None:
        """Keep the chat column at a readable width, centered in the window."""
        try:
            width = page.width or 1280
        except Exception:  # noqa: BLE001
            width = 1280
        available = width - (SIDEBAR_WIDTH if sidebar.visible else 0)
        inner = max(320, min(CONTENT_MAX_WIDTH, int(available) - 72))
        thread.width = inner
        composer_wrap.width = inner
        chat_head.width = inner
        _safe_update()

    body = ft.Row(
        [sidebar, chat_col],
        expand=True,
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )

    page.add(body)
    page.on_resize = _fit_widths

    _render_api_list()
    if chats:
        open_chat(chats[0]["id"])
    else:
        _render_thread(None)
    _render_history()
    _fit_widths()


if __name__ == "__main__":
    ft.run(main)
