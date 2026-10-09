"""Linear pipeline: fetch -> filter (time + AI relevance) -> digest with links.

Relevance is decided by the model, not by substring matching: the topic or
question goes to a cheap selection call that picks the articles by meaning
("спорт" finds hockey and football items) and, when nothing matches, names
the topics actually covered by the fetched news — so the UI can offer them
as clickable chips instead of a dead end. Without usable credentials the
selection falls back to keyword filtering with the scripted chips.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .ai_providers import AIProviderError, get_provider
from .filters import TIME_RANGES, DEFAULT_TIME_RANGE, filter_articles, filter_by_time, parse_tags
from .prompts import attach_links, select_for_prompt
from .scraper import collect_sources, parse_sources
from .topics import suggest_topics

StatusCb = Callable[[str], None]      # status line text
ProgressCb = Callable[[float], None]  # 0.0 .. 1.0

# Russian period phrases for the chat answers (the UI chrome stays English).
_RU_RANGES = {
    "24h": "за последние 24 часа",
    "3d": "за последние 3 дня",
    "7d": "за последние 7 дней",
    "all": "за всё время",
}


class PipelineError(Exception):
    """Pipeline error that should be shown to the user in the status line."""


@dataclass
class PipelineResult:
    summary: str
    stats: str
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)


def _no_match_result(
    *,
    tags: list[str],
    question: str,
    time_range: str,
    articles: list,
    in_range: list,
    stats: str,
    warnings: list[str],
    topic_suggestions: list[str] | None = None,
) -> PipelineResult:
    """A friendly 'nothing matched' answer plus the topics we did find.

    This text is the assistant's reply, so it follows the language of the
    request (Cyrillic request -> Russian message).
    """
    pool = in_range or articles
    suggestions = [s for s in (topic_suggestions or []) if s] or suggest_topics(pool)

    label = ", ".join(tags) or question
    if re.search(r"[а-яё]", label or "", re.I):
        period = _RU_RANGES.get(time_range) or "за выбранный период"
        if label:
            head = f"По запросу **{label}** {period} новостей не нашлось."
        else:
            head = f"Новых новостей {period} нет."
        if suggestions:
            body = (
                f"Проверено {len(articles)} статей ({len(in_range)} в выбранном "
                "периоде) — вот темы, которые ваши источники действительно "
                "освещают. Выберите одну, и я сделаю по ней дайджест:"
            )
        else:
            body = (
                f"Статей получено: {len(articles)}, в выбранном периоде: "
                f"{len(in_range)}. Попробуйте другие теги, расширьте временной "
                "диапазон или оставьте тему пустой для общего дайджеста."
            )
    else:
        range_label = TIME_RANGES.get(time_range, TIME_RANGES[DEFAULT_TIME_RANGE])[0]
        range_label = range_label.lower()
        if label:
            head = f"No news matched **{label}** in {range_label}."
        else:
            head = f"No news found in {range_label}."
        if suggestions:
            body = (
                f"I checked {len(articles)} articles "
                f"({len(in_range)} inside {range_label}) — these are the "
                "topics your sources actually cover right now. Pick one and I'll "
                "build a digest for it:"
            )
        else:
            body = (
                f"Articles fetched: {len(articles)}, in time range: "
                f"{len(in_range)}. Try different tags, widen the time range, or "
                "leave the topic empty for a general digest."
            )

    return PipelineResult(
        summary=f"{head}\n\n{body}",
        stats=stats,
        warnings=warnings,
        suggestions=suggestions,
    )


def run_pipeline(
    settings: dict[str, str],
    on_status: StatusCb,
    on_progress: ProgressCb,
) -> PipelineResult:
    """Full pass: download sources, filter by time and tags, ask the AI."""
    sources = parse_sources(settings.get("sources", ""))
    if not sources:
        raise PipelineError("Add at least one source in the left panel")

    tags = parse_tags(settings.get("tags", ""))
    time_range = settings.get("time_range") or DEFAULT_TIME_RANGE
    range_label = TIME_RANGES.get(time_range, TIME_RANGES[DEFAULT_TIME_RANGE])[0]
    warnings: list[str] = []

    # 1. Fetch ------------------------------------------------------------- #
    on_progress(0.05)
    on_status(f"Fetching news ({len(sources)} sources)...")

    def source_progress(done: int, total: int, src: str) -> None:
        on_status(f"Fetching: {src} ({done}/{total})")
        on_progress(0.05 + 0.45 * done / total)

    articles, errors = collect_sources(sources, on_progress=source_progress)
    warnings.extend(errors)

    # 2. Filtering: time range, then AI relevance --------------------------- #
    on_status("Filtering...")
    on_progress(0.55)
    in_range, undated = filter_by_time(articles, time_range)
    if undated and time_range != "all":
        warnings.append(
            f"{undated} item(s) skipped: no publish date (select 'All time' to include)"
        )

    question = str(settings.get("question") or "").strip()
    matched: list = []
    llm_suggestions: list[str] = []
    if not tags and not question:
        matched = list(in_range)
    elif in_range:
        # The model selects by meaning and, if nothing fits, names the
        # topics the sources actually cover (used as the chips below).
        on_status("Selecting relevant news with AI...")
        on_progress(0.62)
        picked = get_provider(settings).select_articles(in_range, tags, question)
        if picked is not None:
            matched, llm_suggestions = picked
        else:
            matched = filter_articles(in_range, tags)  # offline fallback

    # Which source actually feeds the model (not just how many are configured)
    used = matched or in_range or articles
    per_source = Counter(a.source for a in used)
    breakdown = ", ".join(f"{name} {n}" for name, n in per_source.most_common())
    stats = (
        f"{len(sources)} sources · {breakdown or 'nothing in range'} used · "
        f"{len(articles)} articles · "
        f"{len(in_range)} in {range_label.lower()} · {len(matched)} matched"
        + (
            f" [{', '.join(tags)}]"
            if tags
            else (" (question filter)" if question else " (no topic filter)")
        )
    )

    if not articles:
        details = "; ".join(errors[:4])
        raise PipelineError(f"No source could be loaded. {details}")

    if not matched:
        on_status("Done")
        on_progress(1.0)
        return _no_match_result(
            tags=tags,
            question=question,
            time_range=time_range,
            articles=articles,
            in_range=in_range,
            stats=stats,
            warnings=warnings,
            topic_suggestions=llm_suggestions,
        )

    # 3. AI request --------------------------------------------------------- #
    provider = get_provider(settings)
    on_status(f"Generating summary with {provider.title}...")
    on_progress(0.7)
    options = {
        "temperature": settings.get("gen_temperature", 0.4),
        "length": settings.get("gen_length") or "medium",
        "style": settings.get("gen_style", ""),
        "question": question,
    }
    selected = select_for_prompt(matched)
    try:
        summary = provider.summarize(selected, tags, options)
    except AIProviderError as exc:
        raise PipelineError(str(exc)) from exc

    # 4. Citations [N] -> links to the original articles --------------------- #
    on_status("Adding source links...")
    summary, link_sources = attach_links(summary, selected)

    on_status("Done")
    on_progress(1.0)
    return PipelineResult(
        summary=summary,
        stats=stats,
        warnings=warnings,
        sources=link_sources,
    )
