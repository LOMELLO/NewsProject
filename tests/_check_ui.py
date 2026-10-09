"""Headless structural test for the new UI (no network, temp storage)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flet as ft

from app import chat_store, config, main as main_mod
from app.pipeline import PipelineResult

TMP = Path(tempfile.mkdtemp(prefix="newsproject-"))
config.SETTINGS_FILE = TMP / "settings.json"
chat_store.CHATS_FILE = TMP / "chats.json"

# settings without any API profile -> the no-API branch, no network at all
config.load_settings = lambda: {
    "sources": "https://lenta.ru",
    "time_range": "3d",
    "active_api_id": "",
    "api_profiles": [],
    "gen_length": "medium",
    "gen_style": "",
    "gen_temperature": 0.4,
}


class FakeWindow:
    width = None
    height = None
    min_width = None
    min_height = None
    bgcolor = None


class FakePage:
    def __init__(self):
        self.window = FakeWindow()
        self.controls = []
        self.title = None
        self.width = 1280
        self.updates = 0
        self.dialog = None
        self.show_calls = 0
        self.pop_calls = 0
        self.on_resize = None

    def add(self, *controls):
        self.controls.extend(controls)

    def update(self):
        self.updates += 1

    def show_dialog(self, dlg):
        self.dialog = dlg
        self.show_calls += 1

    def pop_dialog(self):
        self.dialog = None
        self.pop_calls += 1

    def run_thread(self, fn):
        fn()  # synchronous: the pipeline below is mocked


def walk(nodes):
    stack = list(nodes)
    while stack:
        control = stack.pop(0)
        yield control
        for attr in ("controls", "actions"):
            kids = getattr(control, attr, None)
            if kids:
                stack.extend(list(kids))
        inner = getattr(control, "content", None)
        if inner is not None and not isinstance(inner, str):
            stack.append(inner)


def find_button(nodes, label):
    # icon-only controls (the round Send container) carry only a tooltip
    for node in nodes:
        if getattr(node, "tooltip", None) == label and getattr(
            node, "on_click", None
        ):
            return node
    for node in nodes:
        content = getattr(node, "content", None)
        if isinstance(node, (ft.Button, ft.TextButton)) and content == label:
            return node
    raise AssertionError(f"button {label!r} not found")


def texts(nodes):
    return [n for n in nodes if isinstance(n, ft.Text)]


def by_divider(nodes):
    return [
        n for n in nodes
        if isinstance(n, (ft.Divider, ft.VerticalDivider))
    ]


# 1) build ------------------------------------------------------------------
page = FakePage()
main_mod.main(page)
assert page.title == "NewsProject - AI news digest"
assert len(page.controls) == 1, "no top bar: only the body row is on the page"
nodes = list(walk(page.controls))

sidebar = next(n for n in nodes if isinstance(n, ft.Container) and n.width == 300)
assert sidebar.bgcolor == main_mod.BG, sidebar.bgcolor
assert not any(
    (t.value or "") == "Ready" for t in texts(nodes)
), "no status bar"
assert not by_divider(nodes), "no dividers expected"

send_btn = find_button(nodes, "Send")
assert isinstance(send_btn, ft.Container) and send_btn.border_radius == 20, (
    "Send must be a 40x40 circle"
)
assert isinstance(send_btn.content, ft.Icon), "Send is icon-only"
new_chat_btn = find_button(nodes, "New chat")
add_api_btn = find_button(nodes, "Add API")
topic_field = next(
    f for f in nodes if isinstance(f, ft.TextField) and (f.hint_text or "").startswith("Ask about news")
)
assert topic_field.on_submit is not None
print("build OK")

# a question must reach the model, while a topic keeps filtering the articles
assert main_mod._looks_like_question("Что нового в ИИ?")
assert main_mod._looks_like_question("why did the market fall?")
assert not main_mod._looks_like_question("экономика")
assert not main_mod._looks_like_question("AI, спорт")

# 2) send with no API -> friendly error bubble, chat saved -------------------
topic_field.value = "AI"
topic_field.on_submit(None)
nodes = list(walk(page.controls))
mds = [n for n in nodes if isinstance(n, ft.Markdown)]
assert len(mds) == 1, len(mds)
assert "No API selected" in (mds[0].value or ""), mds[0].value
assert mds[0].visible is True
assert send_btn.disabled is False, "the composer must unlock again"

loaded = chat_store.load_chats()
assert len(loaded) == 1, loaded
assert len(loaded[0]["messages"]) == 2, loaded[0]["messages"]
assert loaded[0]["title"] == "AI"
print("no-API send OK; chat saved:", loaded[0]["title"])

# 3) mocked pipeline: answer + topic suggestions -----------------------------
captured_cfg = {}


def fake_pipeline(cfg, on_status, on_progress):
    captured_cfg.clear()
    captured_cfg.update(cfg)
    on_status("Fetching news...")
    on_progress(0.5)
    return PipelineResult(
        summary="Главное: рост [1]\n\n---\n\n**Sources**\n\n- [Новость](https://example.com/1)",
        stats="2 sources · lenta.ru 6, ria.ru 4 used · 10 articles · "
              "5 in last 7 days · 5 matched",
        warnings=["dead source"],
        suggestions=["искусственный интеллект", "рынок"],
    )


main_mod.run_pipeline = fake_pipeline
# an API profile now exists, so the worker branch is taken
config.load_settings = lambda: dict(
    config.DEFAULT_SETTINGS,
    sources="https://lenta.ru",
    api_profiles=[{
        "id": "p1", "name": "Test", "base_url": "https://x/v1", "api_key": "k",
        "model": "m", "status": "unknown", "status_detail": "",
    }],
    active_api_id="p1",
)
page2 = FakePage()
main_mod.main(page2)
nodes2 = list(walk(page2.controls))
field2 = next(
    f for f in nodes2
    if isinstance(f, ft.TextField) and (f.hint_text or "").startswith("Ask about news")
)
field2.value = "topic-that-matches-nothing"
field2.on_submit(None)

nodes2 = list(walk(page2.controls))
mds2 = [n for n in nodes2 if isinstance(n, ft.Markdown)]
assert len(mds2) == 2, len(mds2)  # restored error + the new answer
fresh = mds2[-1]
assert fresh.visible is True
assert "Sources" in (fresh.value or "") and "https://example.com/1" in fresh.value
assert all(m.on_tap_link is not None for m in mds2), "links must be clickable"

chips = [n for n in nodes2 if isinstance(n, ft.Chip)]
labels = [str(c.label) for c in chips]
assert "искусственный интеллект" in labels, labels
assert "рынок" in labels, labels

stats = [t for t in texts(nodes2) if "matched" in (t.value or "")]
assert stats, "stats line expected"
assert any("dead source" in (t.value or "") for t in texts(nodes2)), "warnings"

# suggestion chip re-sends the query
chips[labels.index("рынок")].on_click(None)
nodes2 = list(walk(page2.controls))
user_texts = [t.value for t in texts(nodes2)]
assert "рынок" in user_texts, user_texts
assert not any((t or "").startswith("Topic: ") for t in user_texts), user_texts
print("answer + suggestions OK; user bubbles:", [t for t in user_texts if t])
assert captured_cfg["tags"] == "рынок" and captured_cfg.get("question") == "", captured_cfg

# a real question is passed through instead of being used as a filter tag
field2.value = "Что происходит в ИИ?"
field2.on_submit(None)
assert captured_cfg.get("question") == "Что происходит в ИИ?", captured_cfg
assert captured_cfg["tags"] == "", captured_cfg

# generation options from the composer widgets must reach the pipeline
length_dd = [
    n for n in nodes2
    if isinstance(n, ft.Dropdown)
    and n.options and getattr(n.options[0], "key", None) == "short"
]
assert length_dd, "length dropdown expected"
style_f = [
    n for n in nodes2
    if isinstance(n, ft.TextField)
    and (n.hint_text or "").startswith("e.g. in bullet points")
]
assert style_f, "style field expected"
length_dd[0].value = "detailed"
style_f[0].value = "focus on economics"
field2.value = "ещё одна тема"
field2.on_submit(None)
assert captured_cfg["gen_length"] == "detailed", captured_cfg
assert captured_cfg["gen_style"] == "focus on economics", captured_cfg
assert "gen_temperature" in captured_cfg, captured_cfg
print("generation options OK:", captured_cfg["gen_length"], "/", captured_cfg["gen_style"])

# 4) history: reopen a saved chat ------------------------------------------
loaded2 = chat_store.load_chats()
assert len(loaded2) == 1 and len(loaded2[0]["messages"]) == 10, (
    len(loaded2),
    len(loaded2[0]["messages"]) if loaded2 else 0,
)

new_chat_btn2 = find_button(nodes2, "New chat")
new_chat_btn2.on_click(None)
nodes2 = list(walk(page2.controls))
assert not any(isinstance(n, ft.Markdown) for n in nodes2), "thread must clear"
assert any(
    (t.value or "").startswith("What's in the news") for t in texts(nodes2)
), "empty state back"

history_items = [
    n for n in nodes2
    if isinstance(n, ft.Container)
    and isinstance(n.content, ft.Row)
    and n.on_click is not None
    and isinstance(n.content.controls[0], ft.Column)
    and any(
        isinstance(c, ft.Text) and (c.value or "") == loaded2[0]["title"]
        for c in n.content.controls[0].controls
    )
]
assert history_items, "history entry expected in the sidebar"
history_items[0].on_click(None)  # reopen the saved chat
nodes2 = list(walk(page2.controls))
mds2 = [n for n in nodes2 if isinstance(n, ft.Markdown)]
assert len(mds2) == 5, f"reopened thread must show every answer, got {len(mds2)}"
labels2 = [str(c.label) for c in nodes2 if isinstance(c, ft.Chip)]
assert "рынок" in labels2, labels2
print("history reopen OK")

# 5) rename + delete --------------------------------------------------------
row = history_items[0].content
action_buttons = [c for c in row.controls if isinstance(c, ft.IconButton)]
rename_btn = next(b for b in action_buttons if b.tooltip == "Rename chat")
rename_btn.on_click(None)
assert page2.dialog is not None, "rename dialog must open"
rename_field = next(
    f for f in walk([page2.dialog]) if isinstance(f, ft.TextField) and f.label == "Chat title"
)
rename_field.value = "Renamed chat"
find_button(list(walk([page2.dialog])), "Save").on_click(None)
assert page2.dialog is None
assert chat_store.load_chats()[0]["title"] == "Renamed chat"

delete_btn = next(b for b in action_buttons if b.tooltip == "Delete chat")
delete_btn.on_click(None)
assert chat_store.load_chats() == [], "chat must be deleted"
nodes2 = list(walk(page2.controls))
assert any(
    (t.value or "") == "No chats yet." for t in texts(nodes2)
), "empty history message"
print("rename + delete OK")

# 6) responsive widths ------------------------------------------------------
page2.width = 1000
page2.on_resize(None)
thread = next(
    n for n in walk(page2.controls)
    if isinstance(n, ft.Column) and n.auto_scroll is True
)
assert thread.width == max(320, min(880, 1000 - 300 - 72)), thread.width
print("responsive width OK:", thread.width)

print(f"UI CHECKS PASSED; updates = {page.updates + page2.updates}")
