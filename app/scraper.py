"""News scraping: regular websites (requests + BeautifulSoup) and Telegram (t.me/s)."""

from __future__ import annotations

import json
import re
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .filters import parse_datetime

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}
TIMEOUT = 20
MAX_ARTICLES_PER_SOURCE = 40  # hard per-source cap: fetch everything available, bounded
MAX_TEXT_LEN = 1500

# One t.me/s preview page holds ~15-20 messages; bound pagination requests.
_TELEGRAM_MAX_PAGES = 6

# Feed URLs tried when the page does not advertise one (at most 3 requests).
FEED_CANDIDATES = ("/rss", "/feed", "/rss.xml", "/atom.xml", "/index.xml")


class ScraperError(Exception):
    """A single-source failure (message is shown in the status line)."""


@dataclass
class Article:
    source: str
    url: str
    title: str
    text: str = ""
    published: str = ""  # ISO-8601 publish date if known, else ""


def parse_sources(text: str) -> list[str]:
    """Split the sources box into a list: one per line (or comma-separated)."""
    result: list[str] = []
    for part in re.split(r"[\r\n,;]+", text or ""):
        part = part.strip()
        if part and part not in result:
            result.append(part)
    return result


def is_telegram_source(raw: str) -> bool:
    s = raw.strip().lower()
    return s.startswith("@") or "t.me/" in s or "telegram.me/" in s


def telegram_channel(raw: str) -> str:
    s = raw.strip()
    if s.startswith("@"):
        return s[1:].split("/")[0]
    m = re.search(r"(?:t\.me|telegram\.me)/(?:s/)?([A-Za-z0-9_]+)", s, re.I)
    return m.group(1) if m else ""


def _normalize_url(raw: str) -> str:
    url = raw.strip()
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


_LOGIN_WALL_RE = re.compile(r"var it = (\{.*?\});")


def _fix_encoding(resp: requests.Response) -> requests.Response:
    if not resp.encoding or resp.encoding.upper() == "ISO-8859-1":
        resp.encoding = resp.apparent_encoding  # proper Cyrillic decoding
    return resp


def _wall_meta(body: str) -> dict | None:
    """Parsed `var it = {...}` bootstrap of a Yandex/Dzen SSO shell page."""
    if len(body or "") > 20000:
        return None
    match = _LOGIN_WALL_RE.search(body or "")
    if not match:
        return None
    try:
        meta = json.loads(match.group(1))
    except ValueError:
        return None
    if not isinstance(meta, dict):
        return None
    return meta


def _pass_login_wall(resp: requests.Response) -> requests.Response:
    """Replay Yandex/Dzen-style SSO interstitials (dzen.ru and friends).

    Such sites answer a plain request with a ~3 KB JS shell: ``var it =
    {"host": "...sso...install...", "retpath": "<back url>"}``. The browser
    visits ``host`` (picking up cookies) and is sent back to ``retpath`` —
    only then the real page arrives. One replay is not always enough: the
    server may answer the first ``retpath`` with a fresh shell, while the
    install response has in the meantime set the missing SSO cookies. So we
    repeat the pair until the answer stops being a shell (at most 3 times).
    """
    meta = _wall_meta(resp.text or "")
    if not meta:
        return resp
    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.update(resp.cookies)
    latest = resp
    for _ in range(3):
        install = str(meta.get("host") or "")
        retpath = str(meta.get("retpath") or "")
        if "sso" not in install.lower():
            return latest  # some unrelated `var it = {...}` script
        if not install.startswith(("http://", "https://")):
            return latest
        if not retpath.startswith(("http://", "https://")):
            return latest
        try:
            session.get(install, timeout=TIMEOUT, allow_redirects=True)
            real = session.get(retpath, timeout=TIMEOUT)
            real.raise_for_status()
        except requests.RequestException:
            return latest  # the shell is still better than an exception
        real = _fix_encoding(real)
        latest = real
        meta = _wall_meta(real.text or "")
        if not meta:
            return latest  # a real page (or at least not this shell)
    return latest


def _curl_get(url: str, cause: requests.RequestException) -> requests.Response:
    """Fallback for resets aimed at the plain-python TLS fingerprint.

    t.me resets handshakes from requests/urllib3 while a browser on the
    same IP works fine — the block follows the client fingerprint, not the
    address, so waiting does not help. curl_cffi impersonates a real Chrome
    client (TLS hello + HTTP/2). Optional dependency: when it is missing
    the original connection error propagates unchanged.
    """
    try:
        from curl_cffi import requests as curl_requests
        from requests.structures import CaseInsensitiveDict
        from requests.utils import get_encoding_from_headers
    except ImportError:
        raise cause
    try:
        raw = curl_requests.get(
            url, headers=HEADERS, timeout=TIMEOUT, impersonate="chrome"
        )
    except Exception:  # noqa: BLE001 - the connection error explains more
        raise cause
    resp = requests.Response()
    resp.status_code = raw.status_code
    resp._content = raw.content  # noqa: SLF001 - populate the requests response
    resp.url = getattr(raw, "url", "") or url
    resp.headers = CaseInsensitiveDict(dict(raw.headers.items()))
    resp.encoding = get_encoding_from_headers(resp.headers)
    return resp


def _get(url: str) -> requests.Response:
    last: requests.RequestException | None = None
    for attempt in range(3):  # brief backoff: sites like t.me reset rapid
        try:  # consecutive connections (ConnectionError/Timeout only)
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            break
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
            if attempt < 2:
                _time.sleep(1.5 * (attempt + 1))
    else:
        resp = _curl_get(url, last)  # type: ignore[arg-type]
    resp.raise_for_status()
    resp = _fix_encoding(resp)
    resp = _pass_login_wall(resp)
    return _fix_encoding(resp)


# --------------------------------------------------------------------------- #
# Regular websites
# --------------------------------------------------------------------------- #

def _page_items(soup: BeautifulSoup, page_url: str) -> list[Article]:
    """Universally extract articles from a front page:
    <article> blocks first, then heading links."""
    netloc = urlparse(page_url).netloc
    items: list[Article] = []
    seen: set[str] = set()

    def add(title: str, href: str, snippet: str = "", published: str = "") -> None:
        title = re.sub(r"\s+", " ", title).strip()
        if len(title) < 15:
            return
        url = urljoin(page_url, href).split("#")[0] if href else page_url
        if url in seen:
            return
        seen.add(url)
        items.append(
            Article(source=netloc, url=url, title=title, text=snippet, published=published)
        )

    # 1) <article> containers (WordPress, Medium, modern CMSs)
    for block in soup.find_all("article"):
        h = block.find(["h1", "h2", "h3", "h4"])
        a = block.find("a", href=True)
        if h is None or a is None:
            continue
        paragraphs = [p.get_text(" ", strip=True) for p in block.find_all("p")]
        snippet = " ".join(p for p in paragraphs if len(p) > 20)[:400]
        t = block.select_one("time[datetime]")
        published = (t.get("datetime") or "").strip() if t is not None else ""
        add(h.get_text(" ", strip=True), a.get("href", ""), snippet, published)
        if len(items) >= MAX_ARTICLES_PER_SOURCE:
            return items

    if items:
        return items

    # 2) headings with links (typical for most news sites)
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        if h.find_parent(["nav", "header", "footer", "aside"]) is not None:
            continue
        a = h.find("a", href=True)
        if a is None:
            a = h.find_parent("a", href=True)
        if a is None:
            continue
        add(h.get_text(" ", strip=True), a.get("href", ""))
        if len(items) >= MAX_ARTICLES_PER_SOURCE:
            break

    if items:
        return items

    # 3) generic fallback: links that look like article URLs
    #    (e.g. /20261006/title-1234567890.html on ria.ru and similar)
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if not _looks_like_article_url(href):
            continue
        if a.find_parent(["nav", "header", "footer", "aside"]) is not None:
            continue
        add(a.get_text(" ", strip=True), href)
        if len(items) >= MAX_ARTICLES_PER_SOURCE:
            break

    if items:
        return items

    # 4) last resort: deep same-host links with a sentence-long label —
    #    covers card/JS-ish layouts where nothing else matched.
    for a in soup.find_all("a", href=True):
        if a.find_parent(["nav", "header", "footer", "aside", "form"]) is not None:
            continue
        href = a.get("href", "")
        full = urljoin(page_url, href).split("#")[0]
        parsed = urlparse(full)
        if parsed.scheme not in ("http", "https") or parsed.netloc != netloc:
            continue
        if re.search(r"\.(?:jpe?g|png|gif|svg|webp|pdf|zip|mp4|js|css)(?:$|\?)",
                     parsed.path, re.I):
            continue
        if len([p for p in parsed.path.split("/") if p]) < 2:
            continue
        label = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
        if len(label) < 30:  # nav/category links are short, articles are not
            continue
        add(label, href)
        if len(items) >= MAX_ARTICLES_PER_SOURCE:
            break
    return items


_ARTICLE_URL_RE = (
    re.compile(r"/(?:19|20)\d{6}/"),                 # /20261006/
    re.compile(r"/(?:19|20)\d{2}/\d{1,2}/"),         # /2026/10/
    re.compile(r"-\d{6,}\.html?(?:$|\?|#)"),         # /title-1234567890.html
    re.compile(r"/(?:news|story|article|post|blog)/[^/]+/?", re.I),
    re.compile(r"/\d{5,}/?(?:$|\?|#)"),              # /1234567/
    re.compile(r"\.(?:shtml|stm|news)(?:$|\?|#)", re.I),
)


def _looks_like_article_url(href: str) -> bool:
    return any(p.search(href) for p in _ARTICLE_URL_RE)


def _paragraph_texts(node) -> list[str]:
    """Body paragraphs of an article page.

    Tries real `<p>` blocks first, then class-based bodies used by CMSs that
    render paragraphs as `<div>`s (ria.ru and friends). The group with the
    most text wins.
    """
    groups: list[list[str]] = []
    for selector in (
        "p",
        '[itemprop="articleBody"]',
        '[class*="article__text"]',
        '[class*="article-text"]',
        '[class*="story-text"]',
        '[class*="paragraph"]',
        '[class*="field--name-body"]',
        '[class*="text-block"]',
        '[class*="post-content"]',
        '[class*="entry-content"]',
    ):
        try:
            found = node.select(selector)
        except Exception:  # noqa: BLE001 - a weird selector is not an error
            continue
        texts: list[str] = []
        seen: set[str] = set()
        for element in found:
            text = re.sub(r"\s+", " ", element.get_text(" ", strip=True))
            if len(text) > 30 and text not in seen:
                seen.add(text)
                texts.append(text)
        if texts:
            groups.append(texts)
    if not groups:
        return []
    return max(groups, key=lambda group: sum(len(t) for t in group))


def _looks_like_challenge(html: str) -> bool:
    """True for the tiny 'please run JavaScript' pages anti-bot services serve."""
    text = (html or "")
    head = text[:12000].lower()
    markers = (
        "js-challenge",
        "servicepipe",
        "generateuuid",
        "captcha_frame",
        "checking your browser",
        "enable javascript and cookies",
        '<noscript><meta http-equiv="refresh"',
        "cloudflare-ray-id",
    )
    return any(marker in head for marker in markers)


def _extract_title(soup: BeautifulSoup) -> str:
    """og:title / <title> / first <h1> — used by the sitemap fallback."""
    for sel in (
        'meta[property="og:title"]',
        'meta[name="twitter:title"]',
        'meta[name="title"]',
    ):
        tag = soup.select_one(sel)
        if tag is not None:
            content = (tag.get("content") or "").strip()
            if len(content) >= 12:
                return re.sub(r"\s+", " ", content)
    for sel in ("h1", "title"):
        node = soup.select_one(sel)
        if node is not None:
            text = re.sub(r"\s+", " ", node.get_text(" ", strip=True))
            if len(text) >= 12:
                return text
    return ""


def _fetch_article_details(url: str) -> tuple[str, str, str]:
    """Full article text (first ~1500 chars), publish date and title.

    Returns ("", "", "") when the page cannot be downloaded or is a
    JavaScript challenge instead of an article.
    """
    try:
        resp = _get(url)
    except (requests.RequestException, ValueError):
        return "", "", ""
    body = resp.text or ""
    if _looks_like_challenge(body):
        return "", "", ""
    soup = BeautifulSoup(body, "lxml")
    published = _extract_date(soup)
    title = _extract_title(soup)
    node = None
    for sel in (
        "article",
        "main",
        '[itemprop="articleBody"]',
        '[class*="article-body"]',
        '[class*="article__body"]',
        '[class*="story-body"]',
        '[class*="post-content"]',
        '[class*="entry-content"]',
        '[class*="article__content"]',
        '[class*="article__text"]',
    ):
        found = soup.select_one(sel)
        if found is not None and len(found.get_text(strip=True)) > 200:
            node = found
            break
    if node is None:
        node = soup.body if soup.body is not None else soup
    parts = _paragraph_texts(node)
    if not parts and node is not soup.body and soup.body is not None:
        parts = _paragraph_texts(soup.body)
    text = re.sub(r"\s+", " ", " ".join(parts))[:MAX_TEXT_LEN]
    return text, published, title


# --------------------------------------------------------------------------- #
# RSS / Atom feeds: the most reliable mix of title + link + publish date
# --------------------------------------------------------------------------- #

def _feed_date(raw: str) -> str:
    """Feed date (RFC-822 or ISO-8601) -> ISO-8601 string, else ''."""
    value = (raw or "").strip()
    if not value:
        return ""
    try:  # RSS: "Tue, 06 Oct 2026 19:02:00 +0300"
        dt = parsedate_to_datetime(value)
        if dt is not None:
            return dt.isoformat()
    except (TypeError, ValueError):
        pass
    try:  # Atom: "2026-10-06T19:02:00+03:00"
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.isoformat()
    except ValueError:
        return ""


def _feed_text(entry) -> str:
    """Plain text of <description>/<summary>/<content>/<full-text>, longest wins."""
    wanted = {"description", "summary", "content", "encoded", "full-text"}
    best = ""
    for node in entry.find_all(True):
        local = node.name.split(":")[-1].lower()
        if local not in wanted:
            continue
        raw = node.string if node.string is not None else node.get_text(" ", strip=True)
        raw = (raw or "").strip()
        if not raw:
            continue
        if "<" in raw and ">" in raw:
            raw = BeautifulSoup(raw, "lxml").get_text(" ", strip=True)
        raw = re.sub(r"\s+", " ", raw).strip()
        if len(raw) > len(best):
            best = raw
    return best[:4000] if len(best) >= 40 else ""


def _parse_feed(xml_text: str, feed_url: str) -> list[Article]:
    """RSS 2.0 / RDF / Atom document -> articles (empty list if not a feed)."""
    try:
        soup = BeautifulSoup(xml_text, "xml")
    except Exception:  # noqa: BLE001 - a broken feed is not an error for us
        return []
    entries = soup.find_all("entry") or soup.find_all("item")
    if not entries:
        return []

    netloc = urlparse(feed_url).netloc
    items: list[Article] = []
    seen: set[str] = set()
    for entry in entries:
        if len(items) >= MAX_ARTICLES_PER_SOURCE:
            break
        title_el = entry.find("title")
        title = re.sub(r"\s+", " ", title_el.get_text(" ", strip=True)) if title_el else ""
        if len(title) < 8:
            continue

        link = ""
        link_el = entry.find("link")
        if link_el is not None:
            link = (link_el.get("href") or link_el.get_text(strip=True) or "").strip()
        if not link:
            guid_el = entry.find("guid") or entry.find("id")
            if guid_el is not None and str(guid_el.get_text(strip=True)).startswith("http"):
                link = guid_el.get_text(strip=True).strip()
        url = urljoin(feed_url, link).split("#")[0] if link else feed_url
        if url in seen:
            continue
        seen.add(url)

        published = ""
        for node in entry.find_all(True):
            if node.name.split(":")[-1].lower() in (
                "pubdate", "published", "updated", "date", "issued", "created",
            ):
                published = _feed_date(node.get_text(strip=True))
                if published:
                    break

        items.append(
            Article(
                source=netloc,
                url=url,
                title=title,
                text=_feed_text(entry),
                published=published,
            )
        )
    return items


def _fetch_feed(feed_url: str) -> list[Article]:
    """Download one feed; any failure just means 'no feed here'."""
    try:
        resp = _get(feed_url)
    except (requests.RequestException, ValueError):
        return []
    return _parse_feed(resp.text or "", feed_url)


def _discover_feed(soup: BeautifulSoup, page_url: str) -> str:
    """URL of the <link rel="alternate" type="application/rss+xml"> if any."""
    for link in soup.find_all("link", href=True):
        rel = link.get("rel") or []
        if isinstance(rel, str):
            rel = [rel]
        if "alternate" not in [r.lower() for r in rel]:
            continue
        kind = (link.get("type") or "").lower()
        if any(token in kind for token in ("rss", "atom", "json")):
            return urljoin(page_url, link["href"])
    return ""


def _feeds_from_robots(page_url: str) -> list[str]:
    """Feed URLs advertised in robots.txt (Sitemap lines, rss/feed paths).

    This is how protected sites such as tass.ru publish their feed
    (``Allow: /rss/yandex.xml``) without linking it from the page.
    """
    parsed = urlparse(page_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    try:
        resp = _get(origin + "/robots.txt")
    except (requests.RequestException, ValueError):
        return []
    found: list[str] = []
    for line in (resp.text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        directive, value = line.split(":", 1)
        value = value.strip()
        if not value:
            continue
        directive = directive.strip().lower()
        if directive not in ("sitemap", "allow"):
            continue
        candidate = urljoin(origin + "/", value)
        if not re.search(r"rss|feed|atom", candidate, re.I):
            continue
        if candidate not in found:
            found.append(candidate)
    return found[:4]


# --------------------------------------------------------------------------- #
# JSON-LD (NewsArticle / Article blocks embedded in the listing page)
# --------------------------------------------------------------------------- #

def _jsonld_items(soup: BeautifulSoup, page_url: str) -> list[Article]:
    netloc = urlparse(page_url).netloc
    items: list[Article] = []
    seen: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        kind = node.get("@type") or ""
        kinds = kind if isinstance(kind, list) else [kind]
        kinds = {str(k).lower() for k in kinds}
        interesting = {
            "newsarticle", "article", "reportagenewsarticle", "blogposting",
            "liveblogposting", "webpage",
        }
        if kinds & interesting:
            title = str(node.get("headline") or node.get("name") or "").strip()
            raw_url = node.get("url") or ""
            if isinstance(raw_url, dict):
                raw_url = raw_url.get("@id") or ""
            raw_url = str(raw_url).strip()
            if len(title) >= 15 and raw_url and raw_url not in seen:
                seen.add(raw_url)
                published = str(node.get("datePublished") or "")
                description = str(
                    node.get("description") or node.get("articleBody") or ""
                )
                description = re.sub(r"\s+", " ", description).strip()[:400]
                items.append(
                    Article(
                        source=netloc,
                        url=urljoin(page_url, raw_url).split("#")[0],
                        title=re.sub(r"\s+", " ", title),
                        text=description,
                        published=published,
                    )
                )
        for key in ("@graph", "mainEntity", "itemListElement", "item"):
            if key in node:
                walk(node[key])

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string if script.string is not None else script.get_text()
        raw = (raw or "").strip()
        if not raw:
            continue
        try:
            walk(json.loads(raw))
        except ValueError:
            continue
        if len(items) >= MAX_ARTICLES_PER_SOURCE:
            break
    return items[:MAX_ARTICLES_PER_SOURCE]


# --------------------------------------------------------------------------- #
# Fetching a website: feed -> JSON-LD -> HTML extraction
# --------------------------------------------------------------------------- #

def _looks_like_feed(text: str) -> bool:
    head = (text or "").lstrip("\ufeff \t\r\n")[:2000].lower()
    return "<rss" in head or "<feed" in head or "<rdf" in head


def _enrich(items: list[Article], page_url: str) -> list[Article]:
    """Fetch missing texts/dates in parallel, drop duplicates, newest first."""
    items = items[:MAX_ARTICLES_PER_SOURCE]
    targets = [
        it for it in items
        if len(it.text or "") < 800 and it.url != page_url
    ]
    targets = list({id(t): t for t in targets}.values())
    if targets:
        with ThreadPoolExecutor(max_workers=min(5, len(targets))) as pool:
            futures = {pool.submit(_fetch_article_details, it.url): it for it in targets}
            for fut in as_completed(futures):
                article = futures[fut]
                try:
                    text, published, title = fut.result()
                except Exception:  # noqa: BLE001 - one page must not break the source
                    continue
                if text and len(text) > len(article.text or ""):
                    article.text = text
                if published and not article.published:
                    article.published = published
                if title and not article.title:
                    article.title = title

    unique: dict[str, Article] = {}
    for article in items:
        unique.setdefault(article.url or f"{article.source}|{article.title}", article)
    return _sort_newest(list(unique.values()))


def _sort_newest(items: list[Article]) -> list[Article]:
    """Newest first; articles without a date keep their relative order after."""
    stamped = [(parse_datetime(a.published) if a.published else None, i, a)
               for i, a in enumerate(items)]
    dated = [(dt, i, a) for dt, i, a in stamped if dt is not None]
    undated = [(dt, i, a) for dt, i, a in stamped if dt is None]
    dated.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return [a for _dt, _i, a in dated + undated]


# --------------------------------------------------------------------------- #
# Heavy fallbacks for JS-fronted / protected sites
# --------------------------------------------------------------------------- #

def _feed_fallbacks(page_url: str) -> list[Article]:
    """Conventional feed paths, then feeds advertised in robots.txt."""
    base = page_url if page_url.endswith("/") else page_url + "/"
    for path in FEED_CANDIDATES[:3]:
        items = _fetch_feed(urljoin(base, path))
        if items:
            return items
    for candidate in _feeds_from_robots(page_url)[:2]:
        items = _fetch_feed(candidate)
        if items:
            return items
    return []


def _read_sitemap(url: str) -> tuple[list[str], list[tuple[str, str]]]:
    """One sitemap document -> (child sitemaps, [(loc, lastmod), ...])."""
    try:
        resp = _get(url)
    except (requests.RequestException, ValueError):
        return [], []
    text = resp.text or ""
    if "<urlset" not in text and "<sitemapindex" not in text:
        return [], []
    try:
        soup = BeautifulSoup(text, "xml")
    except Exception:  # noqa: BLE001 - a broken sitemap is not an error
        return [], []
    if soup.find("sitemapindex") is not None:
        children = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
        return children, []
    entries: list[tuple[str, str]] = []
    for node in soup.find_all("url"):
        loc = node.find("loc")
        if loc is None:
            continue
        last = node.find("lastmod")
        entries.append(
            (loc.get_text(strip=True), last.get_text(strip=True) if last else "")
        )
    return [], entries


def _sitemap_articles(page_url: str) -> list[Article]:
    """Last resort: newest URLs from the sitemap, read page by page.

    Helps SPA fronts whose listing is JavaScript-rendered but whose article
    pages are static.
    """
    parsed = urlparse(page_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    roots: list[str] = []
    try:
        resp = _get(origin + "/robots.txt")
        for line in (resp.text or "").splitlines():
            if line.lower().startswith("sitemap:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate and candidate not in roots:
                    roots.append(candidate)
    except (requests.RequestException, ValueError):
        pass
    if not roots:
        roots = [origin + "/sitemap.xml"]

    entries: list[tuple[str, str]] = []
    children: list[str] = []
    for root in roots[:2]:
        kids, own = _read_sitemap(root)
        entries.extend(own)
        children.extend(kids)
    if not entries:
        children = list(
            dict.fromkeys(
                c for c in children if c.startswith(("http://", "https://"))
            )
        )
        children.sort(
            key=lambda c: 0
            if re.search(r"news|article|post|story|blog|index", c, re.I)
            else 1
        )
        for child in children[:2]:
            _kids, own = _read_sitemap(child)
            entries.extend(own)
            if len(entries) >= MAX_ARTICLES_PER_SOURCE:
                break

    if not entries:
        return []
    entries = list(dict.fromkeys(entries))
    entries.sort(key=lambda entry: entry[1] or "", reverse=True)
    urls = [
        loc for loc, _last in entries
        if loc.startswith(("http://", "https://"))
    ][:MAX_ARTICLES_PER_SOURCE]
    if not urls:
        return []

    items: list[Article] = []
    with ThreadPoolExecutor(max_workers=min(5, len(urls))) as pool:
        futures = {pool.submit(_fetch_article_details, u): u for u in urls}
        for future in as_completed(futures):
            article_url = futures[future]
            try:
                text, published, title = future.result()
            except Exception:  # noqa: BLE001
                continue
            if not title or len(title) < 12:
                continue
            items.append(
                Article(
                    source=urlparse(article_url).netloc or parsed.netloc,
                    url=article_url,
                    title=title,
                    text=text,
                    published=published,
                )
            )
            if len(items) >= MAX_ARTICLES_PER_SOURCE:
                break
    return _sort_newest(items)


def fetch_site(raw_url: str) -> list[Article]:
    """Download a site and pick up to 10 recent articles.

    Order of strategies (each one is a fallback for the previous):
    advertised RSS/Atom feed -> JSON-LD news blocks -> `<article>`/heading
    links -> "looks like an article URL" heuristics -> deep same-host links
    -> conventional + robots.txt feeds -> sitemap.
    """
    url = _normalize_url(raw_url)
    if not url:
        raise ScraperError("could not parse the URL")
    try:
        resp = _get(url)
    except requests.RequestException as exc:
        raise ScraperError(
            f"failed to download the page ({exc.__class__.__name__})"
        ) from exc

    body = resp.text or ""

    # a feed pasted directly as a source
    if _looks_like_feed(body):
        items = _parse_feed(body, url)
        if items:
            return _enrich(items, url)

    soup = BeautifulSoup(body, "lxml")

    # 1) feed advertised by the page itself (title + link + date in one go)
    advertised = _discover_feed(soup, url)
    items = _fetch_feed(advertised) if advertised else []
    # 2) JSON-LD news blocks published on the listing page
    if not items:
        items = _jsonld_items(soup, url)
    # 3/4/5) HTML extraction
    if not items:
        items = _page_items(soup, url)
    # 6) conventional feed paths + robots.txt feeds
    if not items:
        items = _feed_fallbacks(url)
    # 7) sitemap: newest article URLs, read page by page
    if not items:
        items = _sitemap_articles(url)

    if not items:
        if _looks_like_challenge(body):
            raise ScraperError(
                "the site is behind a JavaScript anti-bot check; add its RSS "
                "feed URL (often listed in /robots.txt) or another source"
            )
        raise ScraperError(
            "no articles found on the page (the site may be JavaScript-rendered)"
        )
    return _enrich(items, url)


# --------------------------------------------------------------------------- #
# Telegram channels (public preview site t.me/s/channel - no Bot API needed)
# --------------------------------------------------------------------------- #

def _telegram_message(msg, page_url: str, channel: str) -> Article | None:
    """One t.me/s preview message -> Article, or None if not a text post.

    The publish date is read from the date anchor only: video posts carry a
    ``<time class="message_video_duration">`` marker WITHOUT a datetime
    attribute, and it sits in the DOM before the real date.
    """
    text_el = msg.select_one("div.tgme_widget_message_text")
    if text_el is None:
        return None  # skip photo/service messages
    text = re.sub(r"\s+", " ", text_el.get_text(" ", strip=True)).strip()
    if len(text) < 15:
        return None
    time_el = msg.select_one("a.tgme_widget_message_date time")
    if time_el is None:
        time_el = msg.select_one("time[datetime]")
    published = time_el.get("datetime", "") if time_el is not None else ""
    link_el = msg.select_one("a.tgme_widget_message_date")
    msg_url = link_el.get("href", page_url) if link_el is not None else page_url
    title = text if len(text) <= 90 else text[:90].rstrip() + "…"
    return Article(
        source=f"t.me/{channel}",
        url=msg_url,
        title=title,
        text=text,
        published=published,
    )


def fetch_telegram(raw: str) -> list[Article]:
    """The channel's posts, paginating t.me/s backwards (``?before=<id>``).

    One preview page holds ~15-20 messages, so the per-source cap needs
    several pages; stops early on the cap, on a page that adds nothing, or
    after ``_TELEGRAM_MAX_PAGES`` requests (a partial fetch beats an error).
    """
    channel = telegram_channel(raw)
    if not channel:
        raise ScraperError("could not detect the Telegram channel name")
    page_url = f"https://t.me/s/{channel}"

    items: list[Article] = []
    seen: set[str] = set()
    before: int | None = None
    for _page in range(_TELEGRAM_MAX_PAGES):
        if _page:
            _time.sleep(0.8)  # t.me throttles rapid consecutive requests
        url = page_url + (f"?before={before}" if before is not None else "")
        try:
            resp = _get(url)
        except requests.RequestException as exc:
            if items:
                break  # older pages are optional
            raise ScraperError(
                f"failed to download the channel ({exc.__class__.__name__})"
            ) from exc

        soup = BeautifulSoup(resp.text, "lxml")
        added = 0
        earliest: int | None = None
        for msg in soup.select("div.tgme_widget_message"):
            article = _telegram_message(msg, page_url, channel)
            if article is None:
                continue
            match = re.search(r"/(\d+)$", article.url or "")
            if match:
                msg_id = int(match.group(1))
                earliest = msg_id if earliest is None else min(earliest, msg_id)
            if article.url in seen:
                continue
            seen.add(article.url)
            items.append(article)
            added += 1
            if len(items) >= MAX_ARTICLES_PER_SOURCE:
                break
        if len(items) >= MAX_ARTICLES_PER_SOURCE:
            break
        if added == 0 or earliest is None:
            break  # nothing new (channel is shorter than the cap)
        if before is not None and earliest >= before:
            break  # pagination did not move backwards
        before = earliest

    if not items:
        raise ScraperError(
            "channel not found, private, or has no text posts"
        )
    return items


# --------------------------------------------------------------------------- #
# Parallel fetch across all sources
# --------------------------------------------------------------------------- #

def collect_sources(
    sources: list[str],
    on_progress=None,  # (done: int, total: int, source: str) -> None
) -> tuple[list[Article], list[str]]:
    """Download all sources in parallel. Returns (articles, errors)."""
    if not sources:
        return [], ["Source list is empty"]

    def work(src: str) -> list[Article]:
        if is_telegram_source(src):
            return fetch_telegram(src)
        return fetch_site(src)

    articles: list[Article] = []
    errors: list[str] = []
    workers = min(6, len(sources))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, src): src for src in sources}
        for done, fut in enumerate(as_completed(futures), start=1):
            src = futures[fut]
            try:
                articles.extend(fut.result())
            except ScraperError as exc:
                errors.append(f"{src} — {exc}")
            except requests.RequestException as exc:
                errors.append(f"{src} — network error: {exc.__class__.__name__}")
            except Exception as exc:  # one source must not break the whole pipeline
                errors.append(f"{src} — {exc}")
            if on_progress:
                on_progress(done, len(sources), src)

    # one URL may be syndicated by several sources -> keep it once, newest
    # articles first so the prompt budget is spent on fresh news
    unique: dict[str, Article] = {}
    for article in articles:
        unique.setdefault(article.url or f"{article.source}|{article.title}", article)
    return _sort_newest(list(unique.values())), errors




def _extract_date(soup: BeautifulSoup) -> str:
    """Best-effort publish date of a page as an ISO-8601 string ('' if unknown)."""
    for sel in (
        'meta[property="article:published_time"]',
        'meta[itemprop="datePublished"]',
        'meta[name="date"]',
        'meta[name="pubdate"]',
        'meta[name="DC.date.issued"]',
        'meta[property="og:updated_time"]',
    ):
        tag = soup.select_one(sel)
        if tag is not None:
            content = (tag.get("content") or "").strip()
            if content:
                return content
    # JSON-LD: {"datePublished": "2024-05-01T10:00:00+03:00", ...}
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', raw)
        if m:
            return m.group(1)
    time_tag = soup.select_one("time[datetime]")
    if time_tag is not None:
        return (time_tag.get("datetime") or "").strip()
    return ""
