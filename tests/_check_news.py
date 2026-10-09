"""Quick offline checks for the new pipeline pieces (no network, no UI)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import chat_store, prompts, scraper, topics

# the round-trip below must not touch the user's real chats.json
chat_store.CHATS_FILE = Path(tempfile.mkdtemp(prefix="newsproject-")) / "chats.json"

XML = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><title>T</title>
<item><title>Test news about artificial intelligence and markets</title>
<link>https://ex.com/a/1</link>
<pubDate>Tue, 06 Oct 2026 19:02:00 +0300</pubDate>
<description><![CDATA[<p>Some long description of the news item that should be long enough to be kept here.</p>]]></description></item>
<item><title>Second headline about the economy</title><link>/b/2</link>
<description>Another description long enough for the parser to accept it as text body.</description></item>
<item><title>Third one</title><link>https://ex.com/c/3</link></item>
</channel></rss>"""

items = scraper._parse_feed(XML, "https://ex.com/rss")
assert len(items) == 3, items
assert items[0].published.startswith("2026-10-06"), items[0].published
assert "<p>" not in items[0].text, items[0].text
print("feed parse OK:", [(a.title[:20], a.published[:16]) for a in items])

atom = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Atom entry title about energy</title>
<link href="https://ex.com/x/1"/>
<updated>2026-10-05T10:00:00+03:00</updated>
<summary>Short but long enough summary of the atom entry for tests.</summary></entry></feed>"""
a_items = scraper._parse_feed(atom, "https://ex.com/atom.xml")
assert len(a_items) == 1 and a_items[0].url == "https://ex.com/x/1", a_items
assert a_items[0].published.startswith("2026-10-05"), a_items[0].published
print("atom parse OK:", a_items[0].title, a_items[0].published)

# sorting: newest first, undated last
mixed = [
    scraper.Article("s", "u1", "old", published="2026-01-01T00:00:00"),
    scraper.Article("s", "u2", "nodate"),
    scraper.Article("s", "u3", "new", published="2026-10-06T00:00:00+03:00"),
]
ordered = scraper._sort_newest(mixed)
assert [a.url for a in ordered] == ["u3", "u1", "u2"], [a.url for a in ordered]
print("sort OK")

# telegram: a video-duration <time> marker must not shadow the publish date
tg_html = """
<div class="tgme_widget_message_wrap"><div class="tgme_widget_message">
<div class="tgme_widget_message_bubble">
<div class="tgme_widget_message_video js-message_video">
<time class="message_video_duration js-message_video_duration">0:15</time>
</div>
<div class="tgme_widget_message_text js-message_text">Длинный пост про новости: цена топлива
выросла, эксперты обсуждают последствия для всего рынка.</div>
<a class="tgme_widget_message_date" href="https://t.me/ndnews24/123">
<time datetime="2026-10-09T08:42:46+00:00" class="time">08:42</time>
</a></div></div></div>"""
from bs4 import BeautifulSoup as _Soup  # noqa: E402

_msg = _Soup(tg_html, "lxml").select_one("div.tgme_widget_message")
_tg = scraper._telegram_message(_msg, "https://t.me/s/ndnews24", "ndnews24")
assert _tg is not None
assert _tg.published == "2026-10-09T08:42:46+00:00", _tg.published
assert _tg.url == "https://t.me/ndnews24/123"
assert _tg.source == "t.me/ndnews24"
assert len(_tg.title) <= 91 and _tg.text.startswith("Длинный пост")
# the old generic selector grabbed the duration marker and lost the date
assert _msg.select_one("time").get("datetime", "") == ""
print("telegram date OK:", _tg.published)

# JSON-LD
from bs4 import BeautifulSoup  # noqa: E402

html = """
<html><head><script type="application/ld+json">
{"@graph":[{"@type":"NewsArticle","headline":"Big story about the market",
"url":"https://ex.com/n/1","datePublished":"2026-10-06T09:00:00+03:00",
"description":"A description of the big story that is long enough."}]}
</script></head><body></body></html>"""
j = scraper._jsonld_items(Soup := BeautifulSoup(html, "lxml"), "https://ex.com/")
assert len(j) == 1 and j[0].url == "https://ex.com/n/1", j
print("json-ld OK:", j[0].title, j[0].published)

# topic suggestions
pool = [
    scraper.Article("a", "https://ex.com/1", "ИИ научился писать код: индекс роботизации вырос",
                    text="Компании инвестируют в искусственный интеллект и роботов."),
    scraper.Article("b", "https://ex.com/2", "ИИ в медицине: новые системы диагностики",
                    text="Искусственный интеллект помогает врачам ставить диагнозы."),
    scraper.Article("c", "https://ex.com/3", "Рынок труда под давлением ИИ",
                    text="Эксперты обсуждают, как ИИ меняет рынок труда."),
    scraper.Article("d", "https://ex.com/4", "Футбольный клуб сменил тренера",
                    text="Матч закончился поражением 2:1."),
]
suggestions = topics.suggest_topics(pool)
print("suggestions:", suggestions)
assert suggestions, "topics expected"
assert any("ии" in s for s in suggestions), suggestions

# citations -> links
summary = "Главное: рост рынка [1] и новый диагноз [2]. Также [9] без ссылки."
linked, sources = prompts.attach_links(summary, pool[:3])
print("linked:", linked)
assert "[1](https://ex.com/1)" in linked
assert "[2](https://ex.com/2)" in linked
assert "[9]" in linked and "[9](" not in linked
assert "**Sources**" in linked and len(sources) == 2, sources

# no citations at all -> a fallback block with fresh links
linked2, sources2 = prompts.attach_links("Без цитат.", pool[:3])
assert "**Latest news**" in linked2 and len(sources2) == 3, linked2
print("links OK")

# prompt numbering matches the selected articles
selected = prompts.select_for_prompt(pool)
assert selected == pool
prompt = prompts.build_user_prompt(pool, ["ии"], "short", "")
assert "[1] Source:" in prompt and "clickable link" in prompt
print("prompt OK, length:", len(prompt))

# chat store round-trip
chat = chat_store.create_chat(chat_store.title_from("AI, экономика"))
chat_store.append_message(chat, {"role": "user", "text": "AI", "label": "Topic: AI"})
chat_store.append_message(chat, {"role": "assistant", "text": "digest", "stats": "s"})
chat_store.save_chats([chat])
loaded = chat_store.load_chats()
assert loaded and loaded[0]["id"] == chat["id"], loaded
assert len(loaded[0]["messages"]) == 2
assert loaded[0]["title"] == "AI, экономика"
print("chat store OK:", loaded[0]["title"], chat_store.relative_stamp(loaded[0]["updated"]))

print("ALL OFFLINE CHECKS PASSED")
