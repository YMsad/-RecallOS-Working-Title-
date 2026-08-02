"""Tests for the Streamlit UI using streamlit.testing.v1.AppTest."""

from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

from core import database

APP = "app.py"


class FakeClient:
    """Scripted DeepSeek client reused for the UI flow."""

    def __init__(self, questions: list[dict], related=()) -> None:
        self.questions = list(questions)
        self.related = list(related)
        self.i = 0

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def chat(self, messages, **kwargs) -> str:
        user = messages[1]["content"]
        if "判断学习者的回答是否抓住了要点" in user:
            q = self.questions[self.i]
            self.i += 1
            return json.dumps(
                {"is_correct": q["correct"], "feedback": q["feedback"], "hint": q.get("hint")},
                ensure_ascii=False,
            )
        if "参考解释" in user:
            return "参考解释：机会成本是放弃的价值"
        if "concept_title" in user:
            return json.dumps(
                [{"concept_title": c, "relation_text": "都关于选择"} for c in self.related],
                ensure_ascii=False,
            )
        if "每日总结" in user:
            return json.dumps(
                {"breakthrough": "我终于搞懂了机会成本", "tomorrow_hook": "明天再想"},
                ensure_ascii=False,
            )
        if "层追问" in user:
            return f"问题{self.i + 1}"
        raise AssertionError(f"unexpected prompt: {user[:40]}")


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    database.configure(tmp_path / "test.db")
    yield


def make_questions(n: int) -> list[dict]:
    return [{"question": f"Q{i}", "correct": True, "feedback": "对"} for i in range(n)]


def current_step(at: AppTest) -> str:
    return at.session_state.step if "step" in at.session_state else "home"


def click_by_label(at: AppTest, text: str) -> AppTest:
    """Click the button whose label contains ``text`` and re-run."""
    for button in list(at.button):
        if text in (button.label or ""):
            return button.click().run()
    raise AssertionError(f"no button with label containing {text!r}")


def markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_app_boots_to_home() -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert current_step(at) == "home"
    assert "今天想弄懂什么" in markdown_text(at)


def test_app_full_flow(monkeypatch) -> None:
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient(make_questions(4)))
    at = AppTest.from_file(APP, default_timeout=30).run()

    # home -> start a session
    at.text_input[0].input("机会成本")
    at.text_area[0].input("原文：选择意味着放弃")
    at = click_by_label(at, "开始")
    assert not at.exception
    assert current_step(at) == "learning"

    # answer the 4 layer questions
    for _ in range(4):
        at = at.chat_input[0].set_value("答").run()
        assert not at.exception
    assert current_step(at) == "connections"

    # connections -> summary
    at = click_by_label(at, "进入总结")
    assert current_step(at) == "summary"

    # summary: give own words, generate
    at.text_input[0].input("机会成本是放弃的价值")
    at = click_by_label(at, "生成总结")
    assert not at.exception
    assert "summary_result" in at.session_state

    today = database.get_today_summary()
    assert today is not None
    assert today["breakthrough_text"] == "我终于搞懂了机会成本"
    assert database.get_setting("streak") == "1"

    # summary view shows tomorrow hook
    assert "明天再想" in markdown_text(at)


def test_app_history_view(monkeypatch) -> None:
    cid = database.save_concept("机会成本", "原文")
    database.update_concept(cid, mastery="搞懂了")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    assert current_step(at) == "history"
    assert "机会成本" in markdown_text(at)
