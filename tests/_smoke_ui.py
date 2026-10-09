"""Headless smoke test for the chat-style UI (stub Page, real handlers).

Uses the real ``settings.json`` (an existing API profile is edited and
re-checked over the network) but writes chat history into a temp file so the
user's real ``chats.json`` is never touched.
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flet as ft

from app import chat_store
from app.main import BG, USER_BUBBLE_BG, main

chat_store.CHATS_FILE = Path(tempfile.mkdtemp(prefix="newsproject-")) / "chats.json"


class FakeWindow:
    width = None
    height = None
    min_width = None
    min_height = None
    bgcolor = None
    maximized = False


class FakePage:
    def __init__(self):
        self.window = FakeWindow()
        self.controls = []
        self.title = None
        self.width = 1100
        self.updates = 0
        self.dialog = None
        self.show_calls = 0
        self.pop_calls = 0
        self.scroll_calls = []

    def add(self, *controls):
        self.controls.extend(controls)

    def update(self):
        self.updates += 1

    def show_dialog(self, dlg):
        if self.dialog is not None:
            raise RuntimeError("already open")
        self.dialog = dlg
        self.show_calls += 1

    def pop_dialog(self):
        self.dialog = None
        self.pop_calls += 1

    def scroll_to(self, **kw):
        self.scroll_calls.append(kw)


def walk(nodes):
    stack = list(nodes)
    while stack:
        c = stack.pop(0)
        yield c
        for attr in ("controls", "actions"):
            kids = getattr(c, attr, None)
            if kids:
                stack.extend(list(kids))
        inner = getattr(c, "content", None)
        if inner is not None and not isinstance(inner, str):
            stack.append(inner)


def by_type(nodes, cls):
    return [n for n in nodes if isinstance(n, cls)]


def text_values(nodes) -> list[str]:
    """Every string a Text renders: its value plus all span texts."""
    out = []
    for t in by_type(nodes, ft.Text):
        if t.value:
            out.append(str(t.value))
        for sp in t.spans or []:
            if sp.text:
                out.append(str(sp.text))
    return out


def btn_label(b) -> str:
    content = getattr(b, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, ft.Text):
        return str(content.value)
    return str(content)


def find_button(nodes, label: str):
    """Filled/Tonal buttons subclass Button; TextButton does not.

    Icon-only controls (the round Send container) carry no label, so their
    tooltip identifies them.
    """
    for n in nodes:
        if getattr(n, "tooltip", None) == label and getattr(n, "on_click", None):
            return n
    for b in nodes:
        if isinstance(b, (ft.Button, ft.TextButton)) and btn_label(b) == label:
            return b
    raise AssertionError(f"button {label!r} not found")


def wait_until(fn, timeout=150, step=0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(step)
    return False


# 1) build ------------------------------------------------------------------
page = FakePage()
main(page)
assert page.title == "NewsProject - AI news digest", page.title
assert page.theme_mode == ft.ThemeMode.DARK, "dark theme only"
assert page.window.width == 1280 and page.window.min_width == 960

nodes = list(walk(page.controls))
texts = by_type(nodes, ft.Text)
fields = by_type(nodes, ft.TextField)
dropdowns = by_type(nodes, ft.Dropdown)
icons = by_type(nodes, ft.IconButton)

# no permanent bottom status bar anymore
assert not any(
    (t.value or "") == "Ready" for t in texts
), "old status bar text must be gone"
assert not by_type(nodes, ft.ProgressBar), "no progress bar before a request"

# no separator lines anywhere
assert not by_type(nodes, ft.Divider), "no horizontal dividers"
assert not by_type(nodes, ft.VerticalDivider), "no vertical dividers"

# unified button radius: 10 for every real button (Send is a plain 40x40
# circle container and is checked where it is looked up)
buttons = [
    b for b in nodes
    if isinstance(b, (ft.Button, ft.TextButton, ft.IconButton))
    and getattr(b, "style", None) is not None
]
assert buttons, "styled buttons expected"
for b in buttons:
    shape = b.style.shape
    label = btn_label(b) if hasattr(b, "content") else ""
    assert isinstance(shape, ft.RoundedRectangleBorder) and shape.radius == 10, (
        f"{label or type(b).__name__} radius != 10"
    )

# the menu is the same surface as the window: one whole
sidebar = next(n for n in nodes if isinstance(n, ft.Container) and n.width == 300)
assert sidebar.bgcolor == BG, sidebar.bgcolor

# chat input
topic_field = next(
    f for f in fields if (f.hint_text or "").startswith("Ask about news")
)
send_btn = find_button(nodes, "Send")
assert isinstance(send_btn, ft.Container) and send_btn.border_radius == 20, (
    "Send must be a 40x40 circle"
)
assert isinstance(send_btn.content, ft.Icon), "Send is icon-only"

# sidebar: add API + new chat (history section)
add_btn = find_button(nodes, "Add API")
new_chat_btn = find_button(nodes, "New chat")

# migrated API profile is listed with a status label
values = text_values(nodes)
status_labels = {v for v in values if v in ("available", "unavailable", "not checked")}
assert status_labels, "API list must show an availability status"
assert any("GigaChat" in v for v in values), "migrated profile name"

# no top bar: the burger lives in the menu header, the sidebar carries the
# app name (and a second burger appears in the chat only while it is hidden)
assert any(i.tooltip == "Toggle sidebar" for i in icons), "burger button"

print(
    f"build OK: nodes={len(nodes)} fields={len(fields)} "
    f"dropdowns={len(dropdowns)} status={sorted(status_labels)}"
)

# 2) Add API dialog: user names it, tests, cancels -------------------------
add_btn.on_click(None)
assert page.show_calls == 1 and page.dialog is not None
dlg = list(walk([page.dialog]))
assert page.dialog.title.value == "Add API"
name_field = next(f for f in by_type(dlg, ft.TextField) if f.label == "Profile name")
base_url_field = next(f for f in by_type(dlg, ft.TextField) if f.label == "Base URL")
name_field.value = "My test API"
base_url_field.value = ""

# the dialog is provider-agnostic: credentials are shaped by the base URL
assert any(
    f.label == "API key (optional for local models)" for f in by_type(dlg, ft.TextField)
), "one key field for every provider"

# test with empty credentials -> explicit error, status becomes unavailable
test_btn = find_button(dlg, "Test connection")
test_result = next(
    t for t in by_type(dlg, ft.Text)
    if (t.value or "") in ("", "Testing...")
)
test_btn.on_click(None)
assert wait_until(lambda: (test_result.value or "").startswith("OK") is False
                  and (test_result.value or "") != "Testing..."), "test hung"
assert "not set" in test_result.value or "Error" in test_result.value, test_result.value
assert test_result.color == "#ff7575"
print("add-dialog test (error path) OK:", test_result.value[:70])

# cancel -> dialog closed, profile NOT added
find_button(dlg, "Cancel").on_click(None)
assert page.pop_calls == 1 and page.dialog is None
nodes = list(walk(page.controls))
assert not any("My test API" in (t.value or "") for t in by_type(nodes, ft.Text))
print("add-dialog cancel OK")

# 3) Edit existing profile -> Save -> availability check runs ---------------
edit_btn = next(
    i for i in by_type(nodes, ft.IconButton)
    if i.tooltip == "Edit" and getattr(i, "on_click", None)
)
edit_btn.on_click(None)
assert page.dialog is not None and page.dialog.title.value == "Edit API"
dlg = list(walk([page.dialog]))
name_field = next(f for f in by_type(dlg, ft.TextField) if f.label == "Profile name")
assert name_field.value, "edit dialog must be prefilled"
delete_btn = find_button(dlg, "Delete")
assert delete_btn.visible is True

find_button(dlg, "Save").on_click(None)
assert page.dialog is None, "Save must close the dialog"

# background availability check updates the status label
def status_set():
    return {
        v
        for v in text_values(list(walk(page.controls)))
        if v in ("available", "unavailable", "not checked")
    }

assert wait_until(
    lambda: status_set() & {"available", "unavailable"}, timeout=90
), f"status never resolved: {status_set()}"
print("edit + save + background status check OK:", sorted(status_set()))

# 4) send -> thinking placeholder becomes the final answer ------------------
topic_field.value = "AI"
topic_field.on_submit(None)
assert send_btn.disabled is True, "send button must lock while running"

nodes = list(walk(page.controls))
mds = by_type(nodes, ft.Markdown)
assert len(mds) == 1, f"one assistant markdown expected, got {len(mds)}"
md = mds[0]
assert md.visible is False, "markdown hidden while thinking"

# thinking placeholder: ring + live status + progress inside the AI bubble
rings = by_type(nodes, ft.ProgressRing)
assert len(rings) == 1, "thinking spinner visible during the request"
thinking_texts = [
    t.value
    for t in by_type(nodes, ft.Text)
    if (t.value or "").startswith(
        ("Starting", "Fetching", "Filtering", "Selecting", "Generating", "No API")
    )
]
assert thinking_texts, "live status text must be shown in the chat"

# wait for the worker to finish (live GigaChat digest)
assert wait_until(lambda: send_btn.disabled is False, timeout=180), "worker hung"
nodes = list(walk(page.controls))
md = by_type(nodes, ft.Markdown)[0]
assert md.visible is True, "final answer must replace the placeholder"
assert (md.value or "").strip(), "answer must not be empty"
rings = by_type(nodes, ft.ProgressRing)
assert rings and all(r.visible is False for r in rings), "spinner must hide"
stats = [
    t for t in by_type(nodes, ft.Text)
    if "articles" in (t.value or "") or "sources" in (t.value or "")
]
assert stats, "stats line must appear under the answer"
assert any(
    " used · " in (t.value or "") for t in stats
), f"stats must show per-source usage: {[t.value for t in stats]}"
print("send E2E OK; answer starts:", repr((md.value or "")[:60]))

# 5) New chat clears the thread and history --------------------------------
find_button(list(walk(page.controls)), "New chat").on_click(None)
nodes = list(walk(page.controls))
assert by_type(nodes, ft.Markdown) == [], "chat must be cleared"
assert any(
    (t.value or "").startswith("What's in the news") for t in by_type(nodes, ft.Text)
), "empty state back"
assert not any(
    getattr(n, "bgcolor", None) == USER_BUBBLE_BG for n in nodes
), "history cleared: no user bubble left"
print("new chat OK")
print(f"SMOKE OK; UI updates = {page.updates}")


