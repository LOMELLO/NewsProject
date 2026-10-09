"""Article filtering: keyword tags and publish-time range."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only: scraper imports this module back
    from .scraper import Article

# key -> (label shown in UI, timedelta or None for "no time bound")
TIME_RANGES: dict[str, tuple[str, timedelta | None]] = {
    "24h": ("Last 24 hours", timedelta(hours=24)),
    "3d": ("Last 3 days", timedelta(days=3)),
    "7d": ("Last 7 days", timedelta(days=7)),
    "30d": ("Last 30 days", timedelta(days=30)),
    # no date cutoff: everything fetched within the per-source limits,
    # bounded by the prompt budget — "as much context as the model can take"
    "max": ("Maximum context", None),
}
DEFAULT_TIME_RANGE = "3d"
_LEGACY_RANGES = {"all": "max"}  # the removed "All time" key


def resolve_time_range(time_range: str) -> str:
    """Normalize a range key (legacy 'all' -> 'max', unknown -> default)."""
    key = _LEGACY_RANGES.get(time_range, time_range)
    return key if key in TIME_RANGES else DEFAULT_TIME_RANGE


def normalize(text: str) -> str:
    """Lowercase + fold ё/е so «всё» and «все» match the same."""
    return text.lower().replace("ё", "е")


def parse_tags(text: str) -> list[str]:
    """'AI, space; games' -> ['AI', 'space', 'games']."""
    return [t.strip() for t in re.split(r"[,;\n]+", text or "") if t.strip()]


def filter_articles(articles: list[Article], tags: list[str]) -> list[Article]:
    """Keep articles whose title or body contains any of the tags.

    Matching is by word boundary (so 'AI' does not match inside 'said'),
    case-insensitive with ё/е folding. No tags -> everything is kept.
    """
    if not tags:
        return list(articles)

    patterns = [
        re.compile(r"(?<![0-9a-zа-я])" + re.escape(normalize(tag)))
        for tag in tags
    ]

    matched: list[Article] = []
    for article in articles:
        haystack = normalize(f"{article.title} {article.text}")
        if any(p.search(haystack) for p in patterns):
            matched.append(article)
    return matched


def parse_datetime(value: str) -> datetime | None:
    """Parse an ISO-8601 date string into an aware datetime (naive -> UTC).

    Also tolerates compact (ria.ru: ``20261006T1902``) and dotted
    (``06.10.2026 19:02``) formats.
    """
    if not value:
        return None
    v = value.strip()
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        dt = None
        # compact: YYYYMMDDTHHMM[SS]
        m = re.match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})?$", v)
        if m:
            y, mo, d, hh, mm, ss = m.groups()
            try:
                dt = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss or 0))
            except ValueError:
                return None
        else:
            # dotted: DD.MM.YYYY[ HH:MM[:SS]]
            m = re.match(
                r"^(\d{2})\.(\d{2})\.(\d{4})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?$", v
            )
            if m:
                d, mo, y, hh, mm, ss = m.groups()
                try:
                    dt = datetime(
                        int(y), int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0)
                    )
                except ValueError:
                    return None
            else:
                # sloppy ISO: "2024-05-01 12:30" or a bare date
                m = re.match(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)?", v)
                if not m:
                    return None
                try:
                    dt = datetime.fromisoformat(
                        m.group(1) + "T" + (m.group(2) or "00:00:00")
                    )
                except ValueError:
                    return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def filter_by_time(
    articles: list[Article], time_range: str
) -> tuple[list[Article], int]:
    """Keep articles published within the selected range.

    Returns (kept, undated_count). Undated items are skipped when a range is
    active (so the filter cannot silently lie) and kept for 'max'.
    """
    delta = TIME_RANGES[resolve_time_range(time_range)][1]
    if delta is None:
        return list(articles), 0

    cutoff = datetime.now(timezone.utc) - delta
    kept: list[Article] = []
    undated = 0
    for article in articles:
        dt = parse_datetime(article.published)
        if dt is None:
            undated += 1
            continue
        if dt >= cutoff:
            kept.append(article)
    return kept, undated

