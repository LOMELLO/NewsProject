"""Prompt templates for digest generation, plus citation -> link rendering."""

from __future__ import annotations

import re

from .scraper import Article

SYSTEM_PROMPT = (
    "You are an experienced news analyst and editor. "
    "You are given news items collected automatically from different sources. "
    "Answer in the language of the user's request (topic or question); when "
    "there is no request at all, use the language of the news items "
    "themselves. Use only facts from the provided news and never invent "
    "anything. Skip formal preambles such as 'Here is the summary', never send "
    "back policy boilerplate or disclaimers about being an AI - a news digest "
    "is always safe to write - and never simply restate the list of headlines: "
    "compare the items, connect them and say what they mean."
)

MAX_ARTICLE_CHARS = 800
MAX_TOTAL_CHARS = 45000


def reply_language(request_text: str) -> str:
    """'Russian' or 'English' — the language to name in the prompt.

    An explicit instruction ("write in Russian") is followed far more
    reliably than a reference to "the language of the request", so the
    script of what the user typed decides the language outright.
    """
    return "Russian" if re.search(r"[а-яё]", request_text or "", re.I) else "English"


def articles_language(articles: list[Article]) -> str:
    """'Russian' or 'English' by the script majority of the news texts.

    Used for the general digest (no topic, no question): naming the
    language outright is what actually keeps the model from replying in
    English to a Russian news feed.
    """
    sample = " ".join(f"{a.title} {a.text}" for a in articles)[:4000]
    cyrillic = len(re.findall(r"[а-яё]", sample, re.I))
    latin = len(re.findall(r"[a-z]", sample, re.I))
    return "Russian" if cyrillic > latin else "English"

# Answer-length presets: key -> (UI label, target length in characters).
LENGTH_PRESETS: dict[str, tuple[str, int]] = {
    "short": ("Short", 800),
    "medium": ("Medium", 1500),
    "detailed": ("Detailed", 3500),
}
DEFAULT_LENGTH = "medium"

# "[3]" citation, but not "[3](already-a-link)"
_CITATION_RE = re.compile(r"\[(\d{1,3})\](?!\()")


def _block(article: Article, index: int) -> str:
    """One prompt entry; `index` is the number the model cites back as [index]."""
    text = (article.text or article.title).strip()
    return (
        f"[{index}] Source: {article.source}\n"
        f"Title: {article.title.strip()}\n"
        f"Text: {text[:MAX_ARTICLE_CHARS]}"
    )


def select_for_prompt(articles: list[Article]) -> list[Article]:
    """The articles that actually fit into the prompt budget, in order.

    Both the prompt and the citation links are built from this list, so a
    number the model cites always points at the right article.
    """
    selected: list[Article] = []
    total = 0
    for article in articles:
        block = _block(article, len(selected) + 1)
        if total + len(block) > MAX_TOTAL_CHARS:
            break
        total += len(block)
        selected.append(article)
    return selected


def build_user_prompt(
    articles: list[Article],
    tags: list[str],
    length: str = DEFAULT_LENGTH,
    style: str = "",
    question: str = "",
) -> str:
    """Build the user prompt: topic (tags) + the news itself + requirements.

    `question` is the raw text from the composer when it reads like an actual
    question ("what happened to X?") — it is passed through verbatim so the
    model answers the user instead of paraphrasing the headlines.
    """
    topic = ", ".join(tags) if tags else ""
    question = (question or "").strip()

    selected = select_for_prompt(articles)
    blocks = [_block(article, i) for i, article in enumerate(selected, 1)]

    if question:
        task_lines = [
            f"Task: answer the user's question using ONLY the news below: {question}",
            "If the news does not cover it, say so in one short sentence — "
            "never answer from memory.",
        ]
    elif tags:
        task_lines = [
            f"Task: write a short, dense digest strictly about: {topic}.",
            "Include ONLY news items relevant to this topic; skip the rest.",
        ]
    else:
        task_lines = [
            "Task: write one short digest with ONLY the most important "
            "highlights from all of this news - nothing else.",
        ]

    max_chars = LENGTH_PRESETS.get(length, LENGTH_PRESETS[DEFAULT_LENGTH])[1]
    requirements: list[str] = []
    if question or topic:
        language = reply_language(question or topic)
        requirements.append(
            f"- write the WHOLE answer in {language} - the user writes in "
            f"{language}, so never switch to any other language;"
        )
    else:
        language = articles_language(selected)
        requirements.append(
            f"- write the WHOLE answer in {language} - that is the language "
            "of the news below, so never switch to any other language;"
        )
    requirements += [
        "- rely only on the facts from the provided news;",
        "- cite EVERY news item you mention with its number in square brackets, "
        "e.g. [2] - the number turns into a clickable link to the original "
        "article, so never invent numbers and never write a full URL yourself;",
        "- merge related stories;",
        "- structure the answer with short bullet points and bold subheadings;",
        f"- keep it under {max_chars} characters;",
        "- never just retell the raw list of headlines and never repeat the "
        "source list;",
        "- no preambles, disclaimers or apologies - start straight with the "
        "answer;",
        "- if there is no news on the topic, say so directly.",
    ]
    if style.strip():
        requirements.append(f"- additional style instruction: {style.strip()}")

    head = [f"Topic: {topic or '(none - general digest)'}"]
    if question:
        head.append(f"User's question: {question}")

    return "\n".join(
        [
            *head,
            f"Number of news items: {len(blocks)}",
            "",
            "Collected news:",
            "",
            "\n\n".join(blocks),
            "",
            *task_lines,
            "Requirements:",
            *requirements,
        ]
    )


def attach_links(
    summary: str, articles: list[Article]
) -> tuple[str, list[dict]]:
    """Turn `[N]` citations into links and append a **Sources** block.

    Returns (markdown, [{"title", "url", "source"}, ...]). When the model did
    not cite anything, the most recent articles are listed anyway, so every
    digest still carries links to the news it was built from.
    """
    cited: dict[int, Article] = {}

    def _cite(match: re.Match) -> str:
        index = int(match.group(1))
        if 1 <= index <= len(articles):
            article = articles[index - 1]
            if article.url:
                cited[index] = article
                return f"[{index}]({article.url})"
        return match.group(0)

    text = _CITATION_RE.sub(_cite, summary or "")

    picked = [cited[i] for i in sorted(cited)]
    heading = "Sources"
    if not picked:
        # The model cited nothing: still link the freshest articles, but take
        # them round-robin over the sources so one outlet cannot fill the
        # whole block (this is what makes a digest look single-source).
        heading = "Latest news"
        buckets: dict[str, list[Article]] = {}
        for article in articles:
            if article.url:
                buckets.setdefault(article.source, []).append(article)
        picked = []
        while len(picked) < 6 and any(buckets.values()):
            for source in list(buckets):
                bucket = buckets[source]
                if bucket and len(picked) < 6:
                    picked.append(bucket.pop(0))

    sources: list[dict] = []
    seen: set[str] = set()
    lines = ["", "", "---", "", f"**{heading}**", ""]
    for article in picked:
        if not article.url or article.url in seen:
            continue
        seen.add(article.url)
        title = " ".join((article.title or article.url).split())[:140]
        lines.append(f"- [{title}]({article.url}) — {article.source}")
        sources.append(
            {"title": title, "url": article.url, "source": article.source}
        )
    if not sources:
        return text, []
    return text.rstrip() + "\n" + "\n".join(lines) + "\n", sources
