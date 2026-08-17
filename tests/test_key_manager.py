"""V1.0 — 设备指纹 + Cloudflare Worker 临时 Key 分发 + 客户端集成测试（全部隔离网络）。"""

from __future__ import annotations

import json

import httpx
import pytest

from core import database
from core.client import DeepSeekAuthError, DeepSeekClient
from core.config import Settings
from core.device_fingerprint import get_device_fingerprint
from core.key_manager import (
    DailyLimitExceeded,
    KeyManager,
    KeyManagerError,
)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    database.configure(tmp_path / "test.db")
    yield


# ----------------------------------------------------------- 设备指纹


def test_fingerprint_is_64_hex_and_stable() -> None:
    a = get_device_fingerprint()
    b = get_device_fingerprint()
    assert a == b
    assert len(a) == 64
    int(a, 16)  # 必须是合法十六进制


def test_fingerprint_changes_when_component_changes(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.device_fingerprint._get_mac_address", lambda: "11:22:33:44:55:66"
    )
    a = get_device_fingerprint()
    monkeypatch.setattr(
        "core.device_fingerprint._get_mac_address", lambda: "AA:BB:CC:DD:EE:FF"
    )
    b = get_device_fingerprint()
    assert a != b


def test_fingerprint_helpers_degrade_gracefully(monkeypatch) -> None:
    from core.device_fingerprint import _get_disk_serial

    monkeypatch.setattr("core.device_fingerprint._run", lambda *a, **k: "")
    assert _get_disk_serial() == "unknown_disk"
    # 拿不到任何硬件信息时，指纹仍能生成
    fp = get_device_fingerprint()
    assert len(fp) == 64


# ----------------------------------------------------------- KeyManager


def worker_transport(**handlers) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in handlers:
            return handlers[path](request)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def ok_key_body(request: httpx.Request) -> httpx.Response:
    assert request.url.params.get("device")
    return httpx.Response(
        200,
        json={
            "apiKey": "sk-temp-1",
            "expiresIn": 86400,
            "dailyLimit": 20,
            "dailyUsed": 1,
        },
    )


def test_key_manager_fetches_and_caches(tmp_path) -> None:
    transport = worker_transport(**{"/get-key": ok_key_body})
    with KeyManager(
        "https://key.example", cache_dir=tmp_path, device_id="d" * 64, transport=transport
    ) as km:
        assert km.get_api_key() == "sk-temp-1"
        assert km.get_api_key() == "sk-temp-1"  # 第二次命中缓存
    cache = json.loads((tmp_path / "key_cache.json").read_text(encoding="utf-8"))
    assert cache["api_key"] == "sk-temp-1"
    assert cache["device_id"] == "d" * 64
    assert cache["expires_at"] > 0
    # 用量日志已写
    log = (tmp_path / "usage.log").read_text(encoding="utf-8").strip().splitlines()
    assert len(log) == 1
    assert '"daily_used": 1' in log[0]


def test_key_manager_uses_valid_cache_without_network(tmp_path) -> None:
    cache = tmp_path / "key_cache.json"
    import time

    cache.write_text(
        json.dumps(
            {
                "api_key": "sk-cached",
                "expires_at": time.time() + 3600,
                "device_id": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(500)  # 网络不可用时也不应被调用
    )
    with KeyManager(
        "https://key.example", cache_dir=tmp_path, device_id="d" * 64, transport=transport
    ) as km:
        assert km.get_api_key() == "sk-cached"


def test_key_manager_rejects_expired_cache_and_refetches(tmp_path) -> None:
    import time

    (tmp_path / "key_cache.json").write_text(
        json.dumps(
            {"api_key": "sk-stale", "expires_at": time.time() - 1, "device_id": "d" * 64}
        ),
        encoding="utf-8",
    )
    transport = worker_transport(**{"/get-key": ok_key_body})
    with KeyManager(
        "https://key.example", cache_dir=tmp_path, device_id="d" * 64, transport=transport
    ) as km:
        assert km.get_api_key() == "sk-temp-1"


def test_key_manager_raises_on_daily_limit(tmp_path) -> None:
    def limit_body(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "今日学习次数已用尽", "code": "DAILY_LIMIT_EXCEEDED"})

    transport = worker_transport(**{"/get-key": limit_body})
    with KeyManager(
        "https://key.example", cache_dir=tmp_path, device_id="d" * 64, transport=transport
    ) as km:
        with pytest.raises(DailyLimitExceeded):
            km.get_api_key()


def test_key_manager_raises_on_server_error(tmp_path) -> None:
    transport = worker_transport(
        **{"/get-key": lambda request: httpx.Response(500, json={"error": "boom"})}
    )
    with KeyManager(
        "https://key.example", cache_dir=tmp_path, device_id="d" * 64, transport=transport
    ) as km:
        with pytest.raises(KeyManagerError):
            km.get_api_key()


def test_key_manager_raises_on_network_error(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    transport = httpx.MockTransport(handler)
    with KeyManager(
        "https://key.example", cache_dir=tmp_path, device_id="d" * 64, transport=transport
    ) as km:
        with pytest.raises(KeyManagerError):
            km.get_api_key()


def test_key_manager_revoke_all(tmp_path) -> None:
    def revoke_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("secret") == "admin-secret"
        return httpx.Response(
            200,
            json={"success": True, "revoked": True, "version": 2},
        )

    transport = worker_transport(**{"/revoke": revoke_handler})
    with KeyManager(
        "https://key.example", cache_dir=tmp_path, device_id="d" * 64, transport=transport
    ) as km:
        result = km.revoke_all("admin-secret")
    assert result["success"] is True
    assert result["version"] == 2


def test_key_manager_revoke_failure(tmp_path) -> None:
    transport = worker_transport(
        **{"/revoke": lambda request: httpx.Response(401, json={"error": "未授权"})}
    )
    with KeyManager(
        "https://key.example", cache_dir=tmp_path, device_id="d" * 64, transport=transport
    ) as km:
        with pytest.raises(KeyManagerError):
            km.revoke_all("wrong-secret")


# ---------------------------------------------------- 客户端集成（Worker 优先 + 手动兜底）


class _FakeKeyManager:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def get_api_key(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def test_client_uses_worker_key_first(tmp_path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 3}},
        )

    settings = Settings(
        deepseek_api_key="sk-manual",
        recallos_worker_url="https://key.example",
        _env_file=None,
    )
    transport = httpx.MockTransport(handler)
    km = _FakeKeyManager(result="sk-worker")
    with DeepSeekClient(
        settings=settings, transport=transport, key_manager=km
    ) as client:
        client.chat([{"role": "user", "content": "hi"}])
    assert km.calls == 1
    assert seen == ["Bearer sk-worker"]


def test_client_falls_back_to_manual_key_when_worker_fails(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        header = request.headers.get("authorization", "")
        assert header == "Bearer sk-manual"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 3}},
        )

    settings = Settings(
        deepseek_api_key="sk-manual",
        recallos_worker_url="https://key.example",
        _env_file=None,
    )
    km = _FakeKeyManager(error=KeyManagerError("离线"))
    with DeepSeekClient(
        settings=settings, transport=httpx.MockTransport(handler), key_manager=km
    ) as client:
        assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert km.calls == 1


def test_client_falls_back_when_daily_limit_exceeded(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer sk-manual"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 3}},
        )

    settings = Settings(
        deepseek_api_key="sk-manual",
        recallos_worker_url="https://key.example",
        _env_file=None,
    )
    km = _FakeKeyManager(error=DailyLimitExceeded("今日额度用尽"))
    with DeepSeekClient(
        settings=settings, transport=httpx.MockTransport(handler), key_manager=km
    ) as client:
        assert client.chat([{"role": "user", "content": "hi"}]) == "ok"


def test_client_raises_when_no_key_anywhere(monkeypatch) -> None:
    from pathlib import Path as _Path

    monkeypatch.setattr("core.config.CONFIG_FILE", _Path("/nonexistent/config.json"))
    settings = Settings(
        deepseek_api_key="",
        recallos_worker_url="https://key.example",
        _env_file=None,
    )
    km = _FakeKeyManager(error=KeyManagerError("离线"))
    with pytest.raises(DeepSeekAuthError):
        DeepSeekClient(
            settings=settings,
            transport=httpx.MockTransport(lambda r: httpx.Response(200)),
            key_manager=km,
        )