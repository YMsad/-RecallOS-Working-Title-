"""Application settings — loaded from environment variables, the .env file,
and a user-level config file (~/.recallos/config.json)."""

from __future__ import annotations

import json
import sys
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


def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def get_data_dir() -> Path:
    """Data directory for the SQLite database.

    Normal runs keep data under the project (``data/``); when frozen by
    PyInstaller (onefile extracts to a temp dir that vanishes on exit) the
    database would be lost, so it moves to the user profile instead.
    """
    if is_frozen():
        return CONFIG_DIR / "data"
    return PROJECT_ROOT / "data"


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

    # V1.0 — Cloudflare Worker 临时 Key 分发（Worker 优先，手动 Key 兜底）。
    # 配置后，客户端会先从 Worker 取 24h 有效的临时 Key；Worker 不可用/限额
    # 用尽时回退到 deepseek_api_key。留空则完全走原有「手动 Key」路径。
    recallos_worker_url: str = ""

    # HTTP behaviour
    # 30 秒超时：连接/读取/写入任一阶段超时都会抛 DeepSeekNetworkError（不再死等）。
    # 可通过环境变量 REQUEST_TIMEOUT 覆盖。
    request_timeout: float = 30.0
    # 重试次数少一点：只对瞬时网络错误/限流生效，避免「超时×多次重试」把单次失败
    # 放大到几分钟（最坏约 60s 内就能抛错并让用户在 UI 上看到重试入口）。
    max_retries: int = 2
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
