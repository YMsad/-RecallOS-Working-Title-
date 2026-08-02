"""Application settings — loaded from environment variables and the .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is one level above the core/ package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


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
    deepseek_model: str = "deepseek-chat"

    # HTTP behaviour
    request_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    retry_jitter: float = 0.5
    retryable_statuses: frozenset[int] = frozenset({429, 500, 502, 503, 504})


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env on first call)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
