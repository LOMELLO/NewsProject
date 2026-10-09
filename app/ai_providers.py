"""Universal AI client: one OpenAI-compatible ``chat/completions`` flow.

The UI no longer splits connections into GigaChat / YandexGPT / other — a
connection is just a base URL + API key + model. The few provider quirks are
detected automatically from the base URL:

- **GigaChat** (``api.giga.chat`` / ``*.devices.sberbank.ru``): the API key
  field takes the "Authorization key" from the Sber console (base64 of
  ``client_id:client_secret``); an OAuth access token is fetched first and
  cached until it expires.
- **Yandex Cloud** (``llm.api.cloud.yandex.net``): use the OpenAI-compatible
  endpoint ``https://llm.api.cloud.yandex.net/foundationModels/v1`` and put
  the full model URI ``gpt://<folder_id>/<model>`` into the Model field.
  Plain API keys are sent with the ``ApiKey`` scheme, IAM tokens
  (``t1.…`` / ``y0_…``) with ``Bearer``.
- **Anything else** (OpenAI, OpenRouter, DeepSeek, Groq, Ollama…): standard
  ``Bearer`` key, or no key at all for local servers.
"""

from __future__ import annotations

import ast
import base64
import json
import re
import time
import uuid

import requests
import urllib3

from .prompts import LENGTH_PRESETS, SYSTEM_PROMPT, build_user_prompt
from .scraper import Article

AI_TIMEOUT = 120  # seconds for a single AI request
_USER_AGENT = "NewsProject/0.2"

DEFAULT_OPTIONS: dict = {
    "temperature": 0.4,
    "length": "medium",  # key of prompts.LENGTH_PRESETS
    "style": "",         # free-form extra instruction from the user
    "question": "",      # raw composer text when it reads like a question
}

# Semantic pre-selection of articles: the model picks by meaning (so a topic
# like "sport" finds football/hockey items without the literal word) and, when
# nothing matches, proposes the topics to offer as clickable chips.
_SELECT_SYSTEM = (
    "You are a news selection engine. You get a numbered list of news items "
    "(headline plus a short lead) and a user request (a topic or a question). "
    "Choose the items relevant to the request by MEANING, not by literal "
    'word match: a topic like "sport" matches football, hockey or tennis '
    'items even when the word "sport" never appears in them.\n'
    "Reply with STRICT JSON only — no prose, no markdown fences:\n"
    '{"relevant": [<matching item numbers>], '
    '"suggestions": [<5-8 short topic labels of 2-4 words each, in the '
    "language of the request, taken from the list>]}\n"
    'Fill "suggestions" only when "relevant" is empty; otherwise reply with '
    "an empty suggestions list."
)

# GigaChat's server-side content moderation answers with boilerplate instead
# of doing the task (typical for war/politics news batches). It arrives as a
# normal 200-response, so it must be recognized explicitly — otherwise it is
# displayed as if it were the digest. GigaChat uses several variants of the
# wording ("не обладает собственным мнением" vs "не обладают собственным
# мнением — их ответы являются обобщением…"), hence patterns, not substrings.
_MODERATION_PATTERNS = (
    r"не обладае\w* собственным мнением",
    r"разговоры на \w+ темы[^.]{0,40}огранич",
    r"ответ сгенерирован нейросетевой моделью",
    r"обобщением информации, находящейся в открытом доступе",
    r"во избежание неправильного толкования",
)


def _is_moderation_refusal(reply: str) -> bool:
    low = (reply or "").lower()
    return any(re.search(pattern, low) for pattern in _MODERATION_PATTERNS)


def _selection_prompt(articles: list[Article], tags: list[str], question: str) -> str:
    request = question or ("Topic: " + ", ".join(tags))
    lines = [f"Request: {request}", ""]
    for index, article in enumerate(articles, start=1):
        lead = re.sub(r"\s+", " ", article.text or "").strip()[:160]
        lines.append(f"{index}. {article.title}" + (f" — {lead}" if lead else ""))
    return "\n".join(lines)


class AIProviderError(Exception):
    """AI API failure (message is shown to the user)."""


def _http_error(provider: str, resp: requests.Response) -> AIProviderError:
    try:
        data = resp.json()
        msg = data.get("message")
        if not msg:
            err = data.get("error")
            msg = err.get("message") if isinstance(err, dict) else err
        if not msg:
            msg = str(data)[:300]
    except ValueError:
        msg = (resp.text or "")[:300] or "empty response"
    hint = ""
    if resp.status_code in (401, 403):
        hint = " — check your API key"
    elif resp.status_code == 404:
        hint = " — check the model name / base URL"
    elif resp.status_code == 429:
        hint = " — rate limit exceeded"
    return AIProviderError(f"{provider}: HTTP {resp.status_code}{hint}. {msg}")


def normalize_options(options: dict | None) -> dict:
    """Fill missing generation options with defaults and clamp values."""
    merged = dict(DEFAULT_OPTIONS)
    if options:
        merged.update({k: v for k, v in options.items() if v is not None})
    try:
        merged["temperature"] = min(1.0, max(0.0, float(merged["temperature"])))
    except (TypeError, ValueError):
        merged["temperature"] = DEFAULT_OPTIONS["temperature"]
    if merged["length"] not in LENGTH_PRESETS:
        merged["length"] = DEFAULT_OPTIONS["length"]
    merged["style"] = str(merged.get("style") or "").strip()
    merged["question"] = str(merged.get("question") or "").strip()
    return merged


_gigachat_tokens: dict[str, tuple[str, float]] = {}  # api key -> (token, expires_at)


class UniversalProvider:
    """One client for every OpenAI-compatible chat API (GigaChat included)."""

    key = "universal"
    GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    def __init__(self, settings: dict[str, str]) -> None:
        self.settings = settings
        self.title = settings.get("title", "").strip() or "AI API"
        self.base_url = settings.get("base_url", "").strip().rstrip("/")
        self.api_key = settings.get("api_key", "").strip()
        self.model = settings.get("model", "").strip()

    # ----------------------------------------------------------- helpers -- #
    @property
    def _is_gigachat(self) -> bool:
        return "giga.chat" in self.base_url or "sberbank" in self.base_url

    @property
    def _is_yandex(self) -> bool:
        return "yandex" in self.base_url

    def _ensure_creds(self) -> None:
        if not self.base_url:
            raise AIProviderError(
                "Base URL is not set (e.g. https://api.openai.com/v1, "
                "https://api.giga.chat/v1 or http://localhost:11434/v1)"
            )
        if not self.model:
            raise AIProviderError("Model name is not set")

    def _gigachat_basic_credential(self) -> str:
        """Value for the Basic Authorization header on the OAuth endpoint.

        Accepts the ready-made base64 ``client_id:client_secret`` blob from
        the Sber console as well as a plain ``id:secret`` pair.
        """
        try:
            decoded = base64.b64decode(self.api_key, validate=True).decode("utf-8")
        except Exception:
            decoded = ""
        if ":" in decoded and all(32 <= ord(ch) < 127 for ch in decoded):
            return self.api_key
        if ":" in self.api_key:
            return base64.b64encode(self.api_key.encode("utf-8")).decode("ascii")
        raise AIProviderError(
            "GigaChat: paste the Authorization key from developers.sber.ru "
            "into the API key field"
        )

    def _gigachat_token(self) -> str:
        now = time.time()
        cached = _gigachat_tokens.get(self.api_key)
        if cached and cached[1] > now + 60:
            return cached[0]
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            resp = requests.post(
                self.GIGACHAT_OAUTH_URL,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "RqUID": str(uuid.uuid4()),
                    "User-Agent": _USER_AGENT,
                    "Authorization": f"Basic {self._gigachat_basic_credential()}",
                },
                data={"scope": "GIGACHAT_API_PERS", "grant_type": "client_credentials"},
                timeout=30,
                verify=False,  # Sber uses its own CA chain
            )
        except requests.RequestException as exc:
            raise AIProviderError(
                f"GigaChat: failed to obtain a token — {exc.__class__.__name__}"
            ) from exc
        if resp.status_code != 200:
            raise _http_error("GigaChat (oauth)", resp)
        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise AIProviderError("GigaChat: oauth response has no access_token")
        if "expires_at" in data:  # unix timestamp
            expires_at = float(data["expires_at"])
        else:  # lifetime in seconds
            expires_at = now + float(data.get("expires_in", 1700))
        _gigachat_tokens[self.api_key] = (token, expires_at)
        return token

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
        if self._is_gigachat:
            headers["Authorization"] = f"Bearer {self._gigachat_token()}"
            headers["Accept"] = "application/json"
        elif self.api_key:
            if self._is_yandex and not self.api_key.startswith(("t1.", "y0_")):
                headers["Authorization"] = f"ApiKey {self.api_key}"
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # -------------------------------------------------------------- API --- #
    def _completion(
        self, system: str, user: str, *, temperature: float, max_tokens: int
    ) -> str:
        self._ensure_creds()
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=AI_TIMEOUT,
                verify=not self._is_gigachat,
            )
        except requests.RequestException as exc:
            raise AIProviderError(
                f"{self.title}: network error — {exc.__class__.__name__}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise _http_error(f"{self.title} ({url})", resp)
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIProviderError(
                f"{self.title}: unexpected response format: {str(data)[:200]}"
            ) from exc

    def summarize(
        self, articles: list[Article], tags: list[str], options: dict | None = None
    ) -> str:
        opts = normalize_options(options)
        max_chars = LENGTH_PRESETS[opts["length"]][1]
        result = self._completion(
            SYSTEM_PROMPT,
            build_user_prompt(
                articles,
                tags,
                length=opts["length"],
                style=opts["style"],
                question=opts.get("question", ""),
            ),
            temperature=opts["temperature"],
            max_tokens=max(300, max_chars),  # ~1 token per char is a safe ceiling
        ).strip()
        if not result:
            raise AIProviderError(f"{self.title}: the model returned an empty reply")
        if _is_moderation_refusal(result):
            raise AIProviderError(
                f"{self.title}: the provider refused this batch of news "
                "(server-side content moderation). Try a different time "
                "range, other sources, or another API profile."
            )
        return result

    def select_articles(
        self,
        articles: list[Article],
        tags: list[str] | None = None,
        question: str = "",
    ) -> tuple[list[Article], list[str]] | None:
        """Let the model pick the articles relevant to the topic/question.

        Returns ``(matched, suggestions)`` — suggestions are topic chips for
        a "nothing matched" answer, taken from the same list. Returns
        ``None`` when the model cannot be asked (no usable credentials,
        network error, a reply we cannot parse) so the caller can fall back
        to plain keyword filtering.
        """
        pool = list(articles)[:80]  # generous: selection sees the whole feed
        if not pool:
            return [], []
        try:
            raw = self._completion(
                _SELECT_SYSTEM,
                _selection_prompt(pool, tags or [], question),
                temperature=0.0,
                max_tokens=400,
            )
        except AIProviderError:
            return None
        try:
            blob = raw[raw.index("{"): raw.rindex("}") + 1]
            try:
                data = json.loads(blob)
            except ValueError:  # the model may use single quotes
                data = ast.literal_eval(blob)
            relevant = [int(x) for x in data["relevant"]]
            raw_suggestions = data.get("suggestions") or []
        except (ValueError, KeyError, TypeError, SyntaxError):
            return None
        matched = [
            pool[number - 1]
            for number in sorted(set(relevant))
            if 1 <= number <= len(pool)
        ]
        suggestions: list[str] = []
        if isinstance(raw_suggestions, list):
            for item in raw_suggestions:
                text = re.sub(r"\s+", " ", str(item)).strip(" -–—•*«»\"'")
                if 3 <= len(text) <= 48 and text.lower() not in {
                    s.lower() for s in suggestions
                }:
                    suggestions.append(text)
                if len(suggestions) >= 8:
                    break
        return matched, suggestions

    def test_connection(self) -> tuple[str, int]:
        """Make one minimal real request to validate the credentials.

        Returns (reply, elapsed_ms); raises AIProviderError on any failure.
        """
        started = time.monotonic()
        reply = self._completion(
            "You are a connectivity checker. Do not explain anything.",
            "Reply with exactly: OK",
            temperature=0.0,
            max_tokens=16,
        ).strip()
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if not reply:
            raise AIProviderError(f"{self.title}: empty reply from the model")
        return reply, elapsed_ms


def get_provider(settings: dict[str, str]) -> UniversalProvider:
    """Build the provider for a flattened profile/settings dict."""
    return UniversalProvider(settings)
