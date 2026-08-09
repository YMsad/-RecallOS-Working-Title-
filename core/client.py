"""DeepSeek API client — Chat Completions wrapper with retries, timeouts and error handling."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class DeepSeekError(Exception):
    """Base exception for all DeepSeek client errors."""


class DeepSeekAuthError(DeepSeekError):
    """Raised when the API key is missing or rejected (401/403)."""


class DeepSeekRateLimitError(DeepSeekError):
    """Raised when the API rate limit is hit (429)."""


class DeepSeekAPIError(DeepSeekError):
    """Raised for unexpected API errors (any non-2xx other than the handled ones above)."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"DeepSeek API error {status_code}: {message}")


class DeepSeekNetworkError(DeepSeekError):
    """Raised when the request fails at the transport level (timeout / connection)."""


class DeepSeekClient:
    """Thin wrapper around the DeepSeek Chat Completions endpoint.

    Example:
        from core import DeepSeekClient

        with DeepSeekClient() as client:
            reply = client.chat([{"role": "user", "content": "你好"}])
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        **overrides: Any,
    ) -> None:
        self.settings = (settings or get_settings()).model_copy(update=overrides)
        if not self.settings.deepseek_api_key:
            raise DeepSeekAuthError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        self._client = httpx.Client(
            base_url=self.settings.deepseek_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self.settings.request_timeout),
            transport=transport,
        )
        self.usage_records: list[dict[str, Any]] = []
        self.total_usage: dict[str, int] = {}
        self.request_count = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DeepSeekClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ API

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model: str | None = None,
    ) -> str:
        """Send a chat request and return the assistant's text reply."""
        data = self.chat_completion(
            messages, temperature=temperature, max_tokens=max_tokens, model=model
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekError(f"Unexpected response shape: {data}") from exc

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model: str | None = None,
    ) -> dict[str, Any]:
        """POST /chat/completions with retry logic. Returns the parsed JSON body."""
        payload: dict[str, Any] = {
            "model": model or self.settings.deepseek_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        last_error: DeepSeekError | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                body = self._post("/chat/completions", payload)
                self.request_count += 1
                self._record_usage(body)
                return body
            except DeepSeekError as exc:
                if not self._is_retryable(exc):
                    raise
                last_error = exc
                logger.warning(
                    "%s — attempt %d/%d", exc, attempt, self.settings.max_retries
                )
                if attempt < self.settings.max_retries:
                    self._sleep_before_retry(attempt)
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------- internals

    def _record_usage(self, body: dict[str, Any]) -> None:
        """Accumulate per-request token usage from the API response."""
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return
        self.usage_records.append(usage)
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                self.total_usage[key] = self.total_usage.get(key, 0) + value
        logger.info("Token usage: %s", usage)

    def usage_summary(self) -> dict[str, int]:
        """Aggregated token usage across all completed chat requests."""
        summary = {"requests": self.request_count}
        summary.update(self.total_usage)
        return summary

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        logger.debug("POST %s payload=%s", path, payload)
        try:
            response = self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise DeepSeekNetworkError(f"Request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise DeepSeekNetworkError(f"Network error: {exc}") from exc

        status = response.status_code
        logger.debug("Response status=%s", status)
        if status in (401, 403):
            raise DeepSeekAuthError(f"Authentication failed (HTTP {status})")
        if status == 429:
            raise DeepSeekRateLimitError("Rate limit exceeded (HTTP 429)")
        if status >= 400:
            raise DeepSeekAPIError(status, self._extract_error(response))
        try:
            return response.json()
        except ValueError as exc:
            raise DeepSeekError(f"Invalid JSON response: {exc}") from exc

    def _is_retryable(self, error: DeepSeekError) -> bool:
        if isinstance(error, (DeepSeekNetworkError, DeepSeekRateLimitError)):
            return True
        if isinstance(error, DeepSeekAPIError):
            return error.status_code in self.settings.retryable_statuses
        return False

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.settings.retry_backoff * (2 ** (attempt - 1))
        delay += random.uniform(0, self.settings.retry_jitter)
        logger.info("Retrying in %.2fs", delay)
        time.sleep(delay)

    @staticmethod
    def _extract_error(response: httpx.Response) -> str:
        try:
            body = response.json()
            return str(body.get("error", {}).get("message") or body)
        except ValueError:
            return response.text or response.reason_phrase


def ping() -> str:
    """Send a minimal chat request to verify connectivity. Returns the reply."""
    with DeepSeekClient() as client:
        return client.chat(
            [{"role": "user", "content": "请只回复两个字：连通"}],
            max_tokens=10,
        )


if __name__ == "__main__":
    print("Pinging DeepSeek API ...")
    print("Reply:", ping())
