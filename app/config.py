"""Configuration: defaults from .env, user overrides persisted to settings.json.

An API profile is just: name + base_url + api_key + model. The concrete
provider (GigaChat / Yandex / OpenAI-compatible) is detected from the URL
inside ai_providers.UniversalProvider.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_FILE = BASE_DIR / "settings.json"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

DEFAULT_SETTINGS: dict = {
    "sources": "https://lenta.ru\nhttps://ria.ru",
    "tags": "",
    "time_range": "3d",
    "active_api_id": "",
    "api_profiles": [],
    # generation options (consumed by pipeline / prompts)
    "gen_temperature": 0.4,
    "gen_length": "medium",
    "gen_style": "",
    # Legacy flat credentials (.env) - migrated to profiles on first load.
    "gigachat_client_id": os.getenv("GIGACHAT_CLIENT_ID", ""),
    "gigachat_client_secret": os.getenv("GIGACHAT_CLIENT_SECRET", ""),
    "gigachat_model": os.getenv("GIGACHAT_MODEL", "GigaChat-2"),
    "yandex_api_key": os.getenv("YANDEX_API_KEY", ""),
    "yandex_folder_id": os.getenv("YANDEX_FOLDER_ID", ""),
    "yandex_model": os.getenv("YANDEX_MODEL", "yandexgpt-lite/latest"),
    "custom_base_url": os.getenv("CUSTOM_BASE_URL", ""),
    "custom_api_key": os.getenv("CUSTOM_API_KEY", ""),
    "custom_model": os.getenv("CUSTOM_MODEL", ""),
}

GIGACHAT_BASE_URL = "https://api.giga.chat/v1"
YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1"


def new_profile_id() -> str:
    return uuid.uuid4().hex[:8]


def _migrate_legacy_credentials(settings: dict) -> None:
    """Convert flat provider credentials into universal profiles."""
    profiles = settings.get("api_profiles")
    if isinstance(profiles, list) and profiles:
        return
    profiles = []
    gc_secret = str(settings.get("gigachat_client_secret") or "").strip()
    if gc_secret:
        profiles.append(
            {
                "id": new_profile_id(),
                "name": "GigaChat",
                "base_url": GIGACHAT_BASE_URL,
                "api_key": gc_secret,
                "model": str(settings.get("gigachat_model") or "GigaChat-2"),
                "status": "unknown",
                "status_detail": "",
            }
        )
    yx_key = str(settings.get("yandex_api_key") or "").strip()
    yx_folder = str(settings.get("yandex_folder_id") or "").strip()
    if yx_key and yx_folder:
        profiles.append(
            {
                "id": new_profile_id(),
                "name": "YandexGPT",
                "base_url": YANDEX_BASE_URL,
                "api_key": yx_key,
                "model": f"gpt://{yx_folder}/{settings.get('yandex_model') or 'yandexgpt-lite/latest'}",
                "status": "unknown",
                "status_detail": "",
            }
        )
    cu_url = str(settings.get("custom_base_url") or "").strip()
    if cu_url:
        profiles.append(
            {
                "id": new_profile_id(),
                "name": "Custom API",
                "base_url": cu_url,
                "api_key": str(settings.get("custom_api_key") or "").strip(),
                "model": str(settings.get("custom_model") or ""),
                "status": "unknown",
                "status_detail": "",
            }
        )
    settings["api_profiles"] = profiles
    if profiles and not settings.get("active_api_id"):
        settings["active_api_id"] = profiles[0]["id"]


def _migrate_profiles(settings: dict) -> None:
    """Convert legacy per-provider profile fields to the universal format."""
    for p in settings.get("api_profiles") or []:
        if "base_url" in p:  # already universal
            continue
        provider = p.get("provider", "custom")
        fields = p.get("fields") or {}
        if provider == "gigachat":
            p["base_url"] = GIGACHAT_BASE_URL
            p["api_key"] = str(fields.get("gigachat_client_secret") or "")
            p["model"] = str(fields.get("gigachat_model") or "GigaChat-2")
        elif provider == "yandex":
            folder = str(fields.get("yandex_folder_id") or "")
            model = str(fields.get("yandex_model") or "yandexgpt-lite/latest")
            p["base_url"] = YANDEX_BASE_URL
            p["api_key"] = str(fields.get("yandex_api_key") or "")
            p["model"] = f"gpt://{folder}/{model}" if folder else model
        else:
            p["base_url"] = str(fields.get("custom_base_url") or "")
            p["api_key"] = str(fields.get("custom_api_key") or "")
            p["model"] = str(fields.get("custom_model") or "")
        p.pop("provider", None)
        p.pop("fields", None)


def load_settings() -> dict:
    """Defaults (.env) -> overlaid with user settings from settings.json."""
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                settings.update(data)
        except (OSError, ValueError):
            pass  # a broken settings.json must not break startup
    if not isinstance(settings.get("api_profiles"), list):
        settings["api_profiles"] = []
    _migrate_legacy_credentials(settings)
    _migrate_profiles(settings)
    return settings


def save_settings(settings: dict) -> None:
    """Persist current settings (including API profiles) to settings.json."""
    SETTINGS_FILE.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def profile_to_settings(profile: dict) -> dict:
    """Flatten a profile into the keys expected by ai_providers."""
    return {
        "title": profile.get("name") or "AI API",
        "base_url": str(profile.get("base_url") or ""),
        "api_key": str(profile.get("api_key") or ""),
        "model": str(profile.get("model") or ""),
    }
