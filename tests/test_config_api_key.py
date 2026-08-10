"""Tests for the ~/.recallos/config.json API key storage and Settings fallback."""

from __future__ import annotations

import json

import core.config as config
from core.config import (
    Settings,
    get_api_key_from_config,
    get_settings,
    reset_settings_cache,
    save_api_key_to_config,
)


def write_config(monkeypatch, tmp_path, key):
    cfg = tmp_path / ".recallos" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"DEEPSEEK_API_KEY": key}), encoding="utf-8")
    monkeypatch.setattr("core.config.CONFIG_FILE", cfg)


def test_get_api_key_missing_config(tmp_path, monkeypatch) -> None:
    assert get_api_key_from_config() == ""


def test_get_api_key_present_config(tmp_path, monkeypatch) -> None:
    write_config(monkeypatch, tmp_path, "sk-saved")
    assert get_api_key_from_config() == "sk-saved"


def test_save_api_key_roundtrip(tmp_path, monkeypatch) -> None:
    save_api_key_to_config("sk-new")
    assert json.loads(config.CONFIG_FILE.read_text(encoding="utf-8")) == {
        "DEEPSEEK_API_KEY": "sk-new"
    }
    assert get_api_key_from_config() == "sk-new"


def test_settings_falls_back_to_config_file(tmp_path, monkeypatch) -> None:
    write_config(monkeypatch, tmp_path, "sk-saved")
    settings = Settings(_env_file=None)
    assert settings.deepseek_api_key == "sk-saved"


def test_settings_env_key_wins_over_config(tmp_path, monkeypatch) -> None:
    write_config(monkeypatch, tmp_path, "sk-saved")
    settings = Settings(_env_file=None, deepseek_api_key="sk-env")
    assert settings.deepseek_api_key == "sk-env"


def test_get_settings_uses_config_fallback(tmp_path, monkeypatch) -> None:
    write_config(monkeypatch, tmp_path, "sk-saved")
    monkeypatch.setattr("core.config.Settings", lambda: Settings(_env_file=None))
    reset_settings_cache()
    assert get_settings().deepseek_api_key == "sk-saved"
