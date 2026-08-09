"""Tests for the DeepSeek client — mocked transport + optional real connectivity check."""

from __future__ import annotations

import json

import httpx
import pytest

from core import get_settings
from core.client import (
    DeepSeekAPIError,
    DeepSeekAuthError,
    DeepSeekClient,
    DeepSeekError,
    DeepSeekRateLimitError,
)
from core.config import Settings

TEST_SETTINGS = Settings(
    deepseek_api_key="test-key",
    deepseek_base_url="https://api.deepseek.com/v1",
    deepseek_model="deepseek-chat",
    max_retries=3,
    retry_backoff=0.0,
    retry_jitter=0.0,
)


def make_client(handler) -> DeepSeekClient:
    transport = httpx.MockTransport(handler)
    return DeepSeekClient(settings=TEST_SETTINGS, transport=transport)


def ok_body(content: str = "你好") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
        },
    )


def error_body(status: int, message: str = "boom") -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message}})


def test_missing_api_key_raises() -> None:
    with pytest.raises(DeepSeekAuthError):
        DeepSeekClient(settings=Settings(deepseek_api_key=""))


def test_chat_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return ok_body("理解瞬间")

    with make_client(handler) as client:
        assert client.chat([{"role": "user", "content": "hi"}]) == "理解瞬间"


def test_max_tokens_is_sent() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return ok_body()

    with make_client(handler) as client:
        client.chat([{"role": "user", "content": "hi"}], max_tokens=42)
    assert seen[0]["max_tokens"] == 42
    assert seen[0]["model"] == "deepseek-chat"


def test_default_max_tokens_cap_is_sent() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return ok_body()

    with make_client(handler) as client:
        client.chat([{"role": "user", "content": "hi"}])
    assert seen[0]["max_tokens"] == 2000


def test_usage_is_recorded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hi"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "prompt_cache_hit_tokens": 40,
                    "prompt_cache_miss_tokens": 60,
                },
            },
        )

    with make_client(handler) as client:
        client.chat([{"role": "user", "content": "hi"}])
        client.chat([{"role": "user", "content": "hi"}])

    assert len(client.usage_records) == 2
    summary = client.usage_summary()
    assert summary["requests"] == 2
    assert summary["prompt_tokens"] == 200
    assert summary["completion_tokens"] == 100
    assert summary["prompt_cache_hit_tokens"] == 80
    assert summary["total_tokens"] == 300


def test_usage_ignored_when_absent() -> None:
    with make_client(lambda request: ok_body()) as client:
        client.chat([{"role": "user", "content": "hi"}])
    assert client.usage_records == []
    assert client.usage_summary() == {"requests": 1}


def test_retries_then_succeeds_on_429() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return error_body(429)
        return ok_body("最终回复")

    with make_client(handler) as client:
        assert client.chat([{"role": "user", "content": "hi"}]) == "最终回复"
    assert calls["n"] == 3


def test_raises_after_max_retries_on_500() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return error_body(500)

    with make_client(handler) as client:
        with pytest.raises(DeepSeekAPIError):
            client.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == TEST_SETTINGS.max_retries


def test_rate_limit_raised_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_body(429)

    with make_client(handler) as client:
        with pytest.raises(DeepSeekRateLimitError):
            client.chat([{"role": "user", "content": "hi"}])


def test_auth_error_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return error_body(401, "invalid api key")

    with make_client(handler) as client:
        with pytest.raises(DeepSeekAuthError):
            client.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1


def test_non_retryable_http_error_raised_immediately() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return error_body(400, "bad request")

    with make_client(handler) as client:
        with pytest.raises(DeepSeekAPIError):
            client.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1


def test_network_error_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("connection refused")
        return ok_body("恢复")

    with make_client(handler) as client:
        assert client.chat([{"role": "user", "content": "hi"}]) == "恢复"
    assert calls["n"] == 2


def test_unexpected_response_shape_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with make_client(handler) as client:
        with pytest.raises(DeepSeekError):
            client.chat([{"role": "user", "content": "hi"}])


def test_real_connectivity() -> None:
    """Ping DeepSeek for real. Skipped unless DEEPSEEK_API_KEY is configured."""
    settings = get_settings()
    if not settings.deepseek_api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    with DeepSeekClient(settings=settings) as client:
        reply = client.chat(
            [{"role": "user", "content": "请只回复两个字：连通"}], max_tokens=10
        )
        assert reply.strip()
