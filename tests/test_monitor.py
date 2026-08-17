"""V1.0 — 用量监控脚本测试（隔离网络，monkeypatch httpx）。"""

from __future__ import annotations

import json

import httpx
import pytest

from scripts import monitor_usage


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """脚本在 import 时就读取了环境变量，测试前重置关键配置。"""
    monkeypatch.setattr(monitor_usage, "TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setattr(monitor_usage, "TELEGRAM_CHAT_ID", "chat-id")
    monkeypatch.setattr(monitor_usage, "REVOKE_SECRET", "admin-secret")
    monkeypatch.setattr(monitor_usage, "WORKER_URL", "https://key.example")
    yield


def _write_log(path, entries) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_check_usage_no_log_file(tmp_path) -> None:
    result = monitor_usage.check_usage(tmp_path / "missing.log")
    assert result["status"] == "no_log"
    assert result["total_requests"] == 0
    assert result["is_over_threshold"] is False


def test_check_usage_counts_today_only(tmp_path) -> None:
    today = "2026-08-17"
    _write_log(
        tmp_path / "usage.log",
        [
            {"timestamp": f"{today}T10:00:00+0800", "device_id": "dev1", "daily_used": 1},
            {"timestamp": f"{today}T11:00:00+0800", "device_id": "dev2", "daily_used": 1},
            {"timestamp": "2026-08-16T10:00:00+0800", "device_id": "dev3", "daily_used": 1},
            {"timestamp": "garbage-line", "device_id": "x"},
        ],
    )
    result = monitor_usage.check_usage(tmp_path / "usage.log", today=today)
    assert result["status"] == "ok"
    assert result["total_requests"] == 2
    assert result["unique_devices"] == 2
    assert result["total_tokens"] == 2 * 5000
    assert result["is_over_threshold"] is False


def test_check_usage_flags_over_threshold(tmp_path) -> None:
    today = "2026-08-17"
    entries = [
        {"timestamp": f"{today}T10:00:00+0800", "device_id": f"dev{i}", "daily_used": 1}
        for i in range(60)
    ]
    _write_log(tmp_path / "usage.log", entries)
    result = monitor_usage.check_usage(
        tmp_path / "usage.log",
        today=today,
        request_threshold=50,
        token_threshold=1_000_000,
    )
    assert result["total_requests"] == 60
    assert result["is_over_threshold"] is True


def test_send_telegram_alert_posts(monkeypatch) -> None:
    captured = {}

    def fake_post(url, json=None, timeout=10.0):  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(monitor_usage.httpx, "post", fake_post)
    ok = monitor_usage.send_telegram_alert("超啦")
    assert ok is True
    assert "sendMessage" in captured["url"]
    assert captured["json"]["chat_id"] == "chat-id"
    assert "告警" in captured["json"]["text"]


def test_send_telegram_alert_skips_when_unconfigured(monkeypatch) -> None:
    monkeypatch.setattr(monitor_usage, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(monitor_usage, "TELEGRAM_CHAT_ID", "")

    def fake_post(*a, **k):
        raise AssertionError("不应发起请求")

    monkeypatch.setattr(monitor_usage.httpx, "post", fake_post)
    assert monitor_usage.send_telegram_alert("超啦") is False


def test_auto_revoke_success(monkeypatch) -> None:
    def fake_get(url, params=None, timeout=10.0):
        assert "secret=admin-secret" in str(params) or params.get("secret") == "admin-secret"
        return httpx.Response(200, json={"success": True, "revoked": True})

    monkeypatch.setattr(monitor_usage.httpx, "get", fake_get)
    result = monitor_usage.auto_revoke()
    assert result["success"] is True


def test_auto_revoke_requires_config(monkeypatch) -> None:
    monkeypatch.setattr(monitor_usage, "REVOKE_SECRET", "")
    result = monitor_usage.auto_revoke()
    assert "error" in result


def test_auto_revoke_failure_returns_error(monkeypatch) -> None:
    def fake_get(*a, **k):
        return httpx.Response(401, json={"error": "未授权"})

    monkeypatch.setattr(monitor_usage.httpx, "get", fake_get)
    result = monitor_usage.auto_revoke()
    assert "error" in result


def test_main_ok_exits_zero(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(monitor_usage, "DEFAULT_USAGE_LOG", tmp_path / "usage.log")
    with open(tmp_path / "usage.log", "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"timestamp": "2026-08-17T10:00:00+0800", "device_id": "dev1"}
            )
            + "\n"
        )
    code = monitor_usage.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "用量正常" in out