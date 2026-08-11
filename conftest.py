"""Pytest configuration — ensures the project root is importable.

Also patches a confirmed Streamlit AppTest bug (upstream issue #12566,
fixed in PR #16123, not yet in a released pip version): the
``LocalScriptRunner`` ForwardMsgQueue accumulates deltas across internal
reruns (``st.rerun``, view switches), so the parsed element tree keeps
stale widgets whose ids no longer exist in session_state — causing
``KeyError: 'st.session_state has no key "$$ID-...-None"'`` when a run
collects widget states. The app itself is unaffected in a real browser.
"""

from __future__ import annotations

import json

import pytest
from streamlit.runtime.scriptrunner import ScriptRunnerEvent


@pytest.fixture(autouse=True)
def isolated_user_config(tmp_path, monkeypatch):
    """Point ~/.recallos/config.json at a per-test temp file and seed it with a
    valid test key, so the app/CLI boot to their normal flows even when the real
    .env has a blanked key (as happens after manual testing). Tests that exercise
    the key-setup path re-point CONFIG_FILE to an empty temp file themselves.
    """
    from core import config

    config_dir = tmp_path / ".recallos"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"
    config_file.write_text(
        json.dumps({"DEEPSEEK_API_KEY": "test-key"}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)
    config.reset_settings_cache()
    yield


def _install_queue_clear(original_init):
    """Wrap LocalScriptRunner.__init__ to clear stale deltas on rerun."""

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        def clear_stale_deltas(_sender, event, **event_kwargs):
            if event == ScriptRunnerEvent.SCRIPT_STARTED:
                self.forward_msg_queue.clear(
                    retain_lifecycle_msgs=True,
                    fragment_ids_this_run=event_kwargs.get("fragment_ids_this_run"),
                )

        self.on_event.connect(clear_stale_deltas, weak=False)

    return patched_init


def pytest_configure(config):
    from streamlit.testing.v1.local_script_runner import LocalScriptRunner

    LocalScriptRunner.__init__ = _install_queue_clear(LocalScriptRunner.__init__)
