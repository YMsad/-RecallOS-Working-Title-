"""Application settings — loaded from environment variables, the .env file,
and a user-level config file (~/.recallos/config.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is one level above the core/ package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# User-level API key storage (fallback when .env has no DEEPSEEK_API_KEY).
CONFIG_DIR = Path.home() / ".recallos"
CONFIG_FILE = CONFIG_DIR / "config.json"


def get_api_key_from_config() -> str:
    """Return the DeepSeek API key from ~/.recallos/config.json, or ''."""
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    key = data.get("DEEPSEEK_API_KEY", "")
    return key if isinstance(key, str) else ""


def save_api_key_to_config(api_key: str) -> None:
    """Persist the DeepSeek API key to ~/.recallos/config.json."""
    data: dict[str, Any] = {}
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    data["DEEPSEEK_API_KEY"] = api_key
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class Settings(BaseSettings):
    """Runtime configuration for RecallOS."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # DeepSeek API
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-flash"

    # HTTP behaviour
    request_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    retry_jitter: float = 0.5
    retryable_statuses: frozenset[int] = frozenset({429, 500, 502, 503, 504})

    @model_validator(mode="after")
    def _fallback_to_config_file(self) -> "Settings":
        """If .env has no API key, fall back to ~/.recallos/config.json."""
        if not self.deepseek_api_key:
            config_key = get_api_key_from_config()
            if config_key:
                self.deepseek_api_key = config_key
        return self


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env on first call)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Drop the cached Settings so the next call re-reads .env / config file."""
    global _settings
    _settings = None
