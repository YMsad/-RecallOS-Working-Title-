"""API Key 管理 —— 从 Cloudflare Worker 获取临时 Key（V1.0）。

策略：Worker 优先 + 手动 Key 兜底。

- 配置了 ``RECALLOS_WORKER_URL`` 时，优先从 Worker 取临时 Key
  （24h 有效、绑定设备指纹、每设备每日限额），并在本机缓存；
- Worker 不可用 / 每日限额用尽 / 离线时，调用方回退到用户在设置页
  配置的手动 Key（见 ``core/client.py``）。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from core.device_fingerprint import get_device_fingerprint

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".recallos"


class KeyManagerError(Exception):
    """Key 分发失败的基类错误。"""


class DailyLimitExceeded(KeyManagerError):
    """Worker 返回 429 —— 今日额度用尽。"""


class KeyManager:
    """负责从 Cloudflare Worker 获取并缓存临时 API Key。

    Example:
        with KeyManager("https://recallos-key.workers.dev") as km:
            api_key = km.get_api_key()
    """

    def __init__(
        self,
        worker_url: str,
        *,
        cache_dir: Path | str | None = None,
        device_id: str | None = None,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.worker_url = (worker_url or "").rstrip("/")
        if not self.worker_url:
            raise KeyManagerError("RECALLOS_WORKER_URL is not configured")
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_file = self.cache_dir / "key_cache.json"
        self.usage_log_file = self.cache_dir / "usage.log"
        self.device_id = device_id or get_device_fingerprint()
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.worker_url,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "KeyManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --------------------------------------------------------------- public

    def get_api_key(self) -> str:
        """返回可用 Key：优先读缓存，缓存失效才请求 Worker。"""
        cached = self._load_cache()
        if cached and self._cache_valid(cached):
            logger.info(
                "using cached temporary key (expires in %ds)",
                int(cached["expires_at"] - time.time()),
            )
            return cached["api_key"]

        data = self._fetch_from_worker()
        self._cache_key(data["apiKey"], data["expiresIn"])
        self._log_usage(data.get("dailyUsed", 0))
        return data["apiKey"]

    def revoke_all(self, admin_secret: str) -> dict[str, Any]:
        """管理员撤销：让 Worker 拒绝一切新 Key 请求。"""
        try:
            response = self._client.get(
                "/revoke", params={"secret": admin_secret}
            )
        except httpx.TimeoutException as exc:
            raise KeyManagerError(f"Revoke request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise KeyManagerError(f"Cannot reach the key server: {exc}") from exc
        if response.status_code != 200:
            raise KeyManagerError(
                f"Revoke failed (HTTP {response.status_code}): {self._extract_error(response)}"
            )
        return response.json()

    # ------------------------------------------------------------ internals

    def _fetch_from_worker(self) -> dict[str, Any]:
        try:
            response = self._client.get("/get-key", params={"device": self.device_id})
        except httpx.TimeoutException as exc:
            raise KeyManagerError(f"Fetching key timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise KeyManagerError(f"Cannot reach the key server: {exc}") from exc

        if response.status_code == 429:
            raise DailyLimitExceeded(
                self._extract_error(response) or "Daily quota exhausted — come back tomorrow"
            )
        if response.status_code != 200:
            raise KeyManagerError(
                f"Key server returned {response.status_code}: {self._extract_error(response)}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise KeyManagerError(f"Key server returned non-JSON: {exc}") from exc
        if not isinstance(data, dict) or not data.get("apiKey"):
            raise KeyManagerError("Key server did not return an apiKey")
        return data

    def _cache_valid(self, cached: dict[str, Any]) -> bool:
        return (
            bool(cached.get("api_key"))
            and cached.get("device_id") == self.device_id
            and isinstance(cached.get("expires_at"), (int, float))
            and cached["expires_at"] > time.time()
        )

    def _load_cache(self) -> dict[str, Any] | None:
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _cache_key(self, api_key: str, expires_in: int | float) -> None:
        data = {
            "api_key": api_key,
            "expires_at": time.time() + float(expires_in),
            "device_id": self.device_id,
        }
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except OSError as exc:
            logger.warning("failed to write key cache: %s", exc)

    def _log_usage(self, daily_used: int | None) -> None:
        """记录一次取 Key（用于本地用量监控，永远不抛错）。"""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "device_id": self.device_id[:16],
            "daily_used": daily_used or 0,
        }
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.usage_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("failed to write usage log: %s", exc)

    @staticmethod
    def _extract_error(response: httpx.Response) -> str:
        try:
            body = response.json()
            return str(body.get("error") or body)
        except ValueError:
            return response.text[:200] or response.reason_phrase
