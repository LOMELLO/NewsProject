"""Persistent chat history: every conversation lives in ``chats.json``.

A chat is a plain dict::

    {"id": "...", "title": "...", "created": 1.0, "updated": 2.0,
     "messages": [{"role": "user"|"assistant", ...}, ...]}

Messages are appended by the UI (a user request, then the assistant answer
with its stats / suggestions / source links), so reopening a chat from the
sidebar restores the very same thread.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .config import BASE_DIR

CHATS_FILE = BASE_DIR / "chats.json"
MAX_CHATS = 100          # oldest chats fall out of the file
MAX_MESSAGES = 400       # per chat, oldest messages are dropped

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


def _now() -> float:
    return time.time()


def new_chat_id() -> str:
    return uuid.uuid4().hex[:12]


def title_from(text: str, limit: int = 48) -> str:
    """Human-readable chat title from the first user request."""
    clean = " ".join((text or "").split())
    if not clean:
        return "Top highlights"
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def create_chat(title: str) -> dict:
    now = _now()
    return {
        "id": new_chat_id(),
        "title": title or "New chat",
        "created": now,
        "updated": now,
        "messages": [],
    }


def append_message(chat: dict, message: dict) -> dict:
    """Append one message to a chat and bump its activity timestamp."""
    message.setdefault("ts", _now())
    chat.setdefault("messages", []).append(message)
    chat["updated"] = float(message["ts"])
    messages = chat["messages"]
    if len(messages) > MAX_MESSAGES:
        del messages[: len(messages) - MAX_MESSAGES]
    return message


def load_chats() -> list[dict]:
    """All saved chats, newest activity first. Broken files -> empty list."""
    try:
        raw = json.loads(CHATS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = raw.get("chats") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []

    chats: list[dict] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        messages = item.get("messages")
        clean = [
            m
            for m in (messages if isinstance(messages, list) else [])
            if isinstance(m, dict) and m.get("role") in (ROLE_USER, ROLE_ASSISTANT)
        ]
        created = item.get("created") or item.get("updated") or _now()
        item["messages"] = clean
        item["title"] = str(item.get("title") or "Chat")
        item["created"] = float(created)
        item["updated"] = float(item.get("updated") or created)
        chats.append(item)

    chats.sort(key=lambda c: c["updated"], reverse=True)
    return chats[:MAX_CHATS]


def save_chats(chats: list[dict]) -> None:
    """Atomically persist the whole history (a crash must not lose it)."""
    payload = {"chats": chats[:MAX_CHATS]}
    tmp = CHATS_FILE.with_name(CHATS_FILE.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(CHATS_FILE)


def relative_stamp(ts: float, now: float | None = None) -> str:
    """'just now' / '5 min' / 'yesterday' / '12.03' label for the sidebar."""
    if not ts:
        return ""
    delta = max(0.0, (now or _now()) - float(ts))
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)} h ago"
    if delta < 172800:
        return "yesterday"
    if delta < 604800:
        return f"{int(delta // 86400)} d ago"
    stamp = time.localtime(float(ts))
    return time.strftime("%d.%m.%Y", stamp)
