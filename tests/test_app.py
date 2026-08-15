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
        if "复习日" in user:
            return "复习问题：机会成本是什么？"
        if "我不懂" in user:
            return "大白话：机会成本就是你放弃的那个次优选择"
        if "预热" in user:
            return "用一句话说，机会成本就是你为了得到A而放弃的B。"
        if "降维" in user:
            return "简化后的问题"
        if "换个角度" in user:
            return "换个角度的问题"
        if "开场第一个问题" in user:
            return f"开场问题{self.i + 1}"
        if "层追问" in user:
            return f"问题{self.i + 1}"
        raise AssertionError(f"unexpected prompt: {user[:40]}")


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    database.configure(tmp_path / "test.db")
    yield


@pytest.fixture
def configured_app(monkeypatch, tmp_path):
    """Give the app a valid API key so it boots to home (not the setup page)."""
    from core.config import Settings

    monkeypatch.setattr("core.config.Settings", lambda: Settings(deepseek_api_key="test-key", _env_file=None))
    from core import config

    config.reset_settings_cache()
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


def test_app_boots_to_home(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert current_step(at) == "home"
    assert "今天想弄懂什么" in markdown_text(at)


def test_app_shows_key_setup_when_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("core.config.CONFIG_FILE", tmp_path / "cfg.json")
    monkeypatch.setattr("core.config.CONFIG_DIR", tmp_path)
    from core.config import Settings

    monkeypatch.setattr("core.config.Settings", lambda: Settings(_env_file=None))
    from core import config

    config.reset_settings_cache()
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert "请配置 DeepSeek API Key" in markdown_text(at)
    assert len(at.text_input) == 1
    assert at.text_input[0].label == "DeepSeek API Key"


def test_app_save_key_returns_to_home(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "cfg.json"
    monkeypatch.setattr("core.config.CONFIG_FILE", cfg)
    monkeypatch.setattr("core.config.CONFIG_DIR", tmp_path)
    from core.config import Settings

    monkeypatch.setattr("core.config.Settings", lambda: Settings(_env_file=None))
    from core import config

    config.reset_settings_cache()
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("sk-app-saved")
    at = click_by_label(at, "保存")
    assert not at.exception
    assert "请配置 DeepSeek API Key" not in markdown_text(at)
    assert config.get_api_key_from_config() == "sk-app-saved"


def test_app_full_flow(monkeypatch, configured_app) -> None:
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


def test_app_history_view(monkeypatch, configured_app) -> None:
    cid = database.save_concept("机会成本", "原文")
    database.update_concept(cid, mastery="搞懂了")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    assert current_step(at) == "history"
    assert "机会成本" in markdown_text(at)
    # 按掌握度分组展示
    assert "✅ 搞懂了" in markdown_text(at)


def test_app_mode_toggle_on_home(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert len(at.toggle) == 1
    assert "有基础" in at.toggle[0].label


def test_app_learning_flow_uses_dynamic_opening(monkeypatch, configured_app) -> None:
    fake = FakeClient(make_questions(4))
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("机会成本")
    at.text_area[0].input("原文：选择意味着放弃")
    at = click_by_label(at, "开始")
    assert not at.exception
    assert current_step(at) == "learning"
    assert "开场问题" in markdown_text(at)


def test_app_explain_button(monkeypatch, configured_app) -> None:
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient(make_questions(4)))
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("机会成本")
    at.text_area[0].input("原文：选择意味着放弃")
    at = click_by_label(at, "开始")
    at = click_by_label(at, "我不懂")
    assert not at.exception
    assert "我换个说法" in markdown_text(at)


def test_app_single_assistant_bubble_per_answer(monkeypatch, configured_app) -> None:
    """每个回答只产生一个 AI 气泡（反馈与下一问合并），不会出现两个气泡。"""
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient(make_questions(4)))
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("机会成本")
    at.text_area[0].input("原文：选择意味着放弃")
    at = click_by_label(at, "开始")
    at = at.chat_input[0].set_value("答").run()
    assert not at.exception
    msgs = at.session_state["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["assistant", "user", "assistant"]
    assert "✓ 对" in msgs[2]["text"]
    assert "问题2" in msgs[2]["text"]


def test_app_warmup_button_shows_prewarm_text(monkeypatch, configured_app) -> None:
    fake = FakeClient(make_questions(4))
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("机会成本")
    at.text_area[0].input("原文：选择意味着放弃")
    at = at.run()
    at = click_by_label(at, "预热")
    assert not at.exception
    assert any(
        "用一句话说，机会成本就是你为了得到A而放弃的B。" in m.value
        for m in at.info
    )


def test_app_warmup_prepended_to_first_message(monkeypatch, configured_app) -> None:
    fake = FakeClient(make_questions(4))
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("机会成本")
    at.text_area[0].input("原文：选择意味着放弃")
    at = at.run()
    at = click_by_label(at, "预热")
    at = click_by_label(at, "开始")
    assert not at.exception
    assert current_step(at) == "learning"
    assert "用一句话说" in markdown_text(at)
    assert "开场问题" in markdown_text(at)


def test_app_reconfigure_dialog_opens(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    at = click_by_label(at, "重新配置 API Key")
    assert not at.exception
    assert current_step(at) == "reconfigure"
    assert any(t.label == "新的 DeepSeek API Key" for t in at.text_input)
    assert any("保存并重新加载" in (b.label or "") for b in at.button)


def test_app_reconfigure_saves_new_key_and_returns_home(configured_app) -> None:
    from core import config

    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "重新配置 API Key")
    key_input = next(t for t in at.text_input if t.label == "新的 DeepSeek API Key")
    key_input.input("sk-new-key-123")
    at = at.run()
    at = click_by_label(at, "保存并重新加载")
    assert not at.exception
    assert config.get_api_key_from_config() == "sk-new-key-123"
    assert current_step(at) == "home"


# ------------------------------------------------------------------ V0.2.2


def _seed_due_concept() -> int:
    """Create a concept that is due for review today."""
    from datetime import date

    cid = database.save_concept("机会成本", "原文：选择意味着放弃")
    database.update_concept(cid, mastery="搞懂了", next_review_date=date.today().isoformat())
    return cid


def test_app_review_entry_shown_on_home(configured_app) -> None:
    _seed_due_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert any("今日复习" in (b.label or "") for b in at.button)


def test_app_review_entry_hidden_when_nothing_due(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert not any("今日复习" in (b.label or "") for b in at.button)


def test_app_review_list_shows_due_concepts(configured_app) -> None:
    _seed_due_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "今日复习")
    assert not at.exception
    assert current_step(at) == "review_list"
    assert "机会成本" in markdown_text(at)
    assert any("开始复习" in (b.label or "") for b in at.button)


def test_app_review_mode_asks_tomorrow_hook_and_passes(monkeypatch, configured_app) -> None:
    from core import review as review_module

    cid = _seed_due_concept()
    database.save_daily_summary(cid, "我终于搞懂了", "明天再想")
    fake = FakeClient([{"question": "Q", "correct": True, "feedback": "对"}] * 3)
    monkeypatch.setattr(review_module, "DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "今日复习")
    at = click_by_label(at, "开始复习")
    assert not at.exception
    assert current_step(at) == "review"
    assert "明天再想" in markdown_text(at)

    at = at.chat_input[0].set_value("答").run()
    assert not at.exception
    assert "对" in markdown_text(at)
    assert any("复习通过" in (s.value or "") for s in at.success)


def test_app_review_mode_three_failures_needs_relearn(monkeypatch, configured_app) -> None:
    from core import review as review_module

    cid = _seed_due_concept()
    database.save_daily_summary(cid, "我终于搞懂了", "明天再想")
    fake = FakeClient([{"question": "Q", "correct": False, "feedback": "再想想"}] * 3)
    monkeypatch.setattr(review_module, "DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "今日复习")
    at = click_by_label(at, "开始复习")
    assert not at.exception
    for _ in range(3):
        at = at.chat_input[0].set_value("错").run()
    assert not at.exception
    assert "重新学" in markdown_text(at)
    assert any("重新学" in (w.value or "") for w in at.warning)
    assert database.get_concept(cid)["mastery"] == "学习中"


def test_app_connection_bidirectional_jump(monkeypatch, configured_app) -> None:
    cid_a = database.save_concept("机会成本", "原文A")
    cid_b = database.save_concept("沉没成本", "原文B")
    database.save_connection(cid_a, cid_b, "都关于选择")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    assert current_step(at) == "history"
    assert any("去往" in (b.label or "") for b in at.button)
    at = click_by_label(at, "去往")
    assert not at.exception
    assert current_step(at) == "concept_detail"
    assert "沉没成本" in markdown_text(at)


def test_app_connection_long_text_preserved(monkeypatch, configured_app) -> None:
    cid_a = database.save_concept("机会成本", "原文A")
    cid_b = database.save_concept("沉没成本", "原文B")
    database.save_connection(cid_a, cid_b, "都关于选择")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    # 长关系说明应完整显示，而不是被截断成单行
    assert "都关于选择" in markdown_text(at)


# --------------------------------------------------------------- usage stats


def test_app_usage_stats_page(configured_app) -> None:
    database.save_usage_log(model="deepseek-chat", prompt_tokens=100,
                            completion_tokens=50, total_tokens=150, cost=0.001)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "用量统计")
    assert not at.exception
    assert current_step(at) == "usage_stats"
    assert "用量统计" in markdown_text(at)
    # 累计/本月/今日都应有 Token 数字（metric 的 value）
    metric_values = [m.value for m in at.metric]
    assert len(metric_values) == 3
    assert any("150" in (v or "") for v in metric_values)


def test_app_usage_stats_empty(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "用量统计")
    assert not at.exception
    assert current_step(at) == "usage_stats"
    assert any("还没有用量数据" in (i.value or "") for i in at.info)


# ---------------------------------------------------------------- V0.2.3


def _seed_unfinished_concept() -> int:
    cid = database.save_concept("机会成本", "原文：选择意味着放弃")
    database.save_qa(cid, "它关注过去还是未来？", "未来", True)
    database.save_qa(cid, "和沉没成本的区别？", "一个看过去", False, hint_used=True)
    return cid


def test_continue_learning_entry(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not any("继续学习" in (b.label or "") for b in at.button)

    _seed_unfinished_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert any("继续学习：机会成本" in (b.label or "") for b in at.button)


def test_continue_learning_restores_messages(configured_app) -> None:
    cid = _seed_unfinished_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "继续学习")
    assert not at.exception
    assert current_step(at) == "learning"

    session = at.session_state["session"]
    assert session.concept_id == cid

    msgs = at.session_state["messages"]
    assert len(msgs) == 4
    assert msgs[0] == {"role": "assistant", "text": "它关注过去还是未来？"}
    assert msgs[1] == {"role": "user", "text": "未来"}
    assert msgs[2] == {"role": "assistant", "text": "和沉没成本的区别？"}
    assert msgs[3] == {"role": "user", "text": "一个看过去"}
    assert "它关注过去还是未来？" in markdown_text(at)


def test_app_delete_concept_flow(configured_app) -> None:
    cid = database.save_concept("机会成本", "原文")
    database.save_qa(cid, "Q?", "A", True)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    at = click_by_label(at, "✕")
    assert not at.exception
    assert any("确认删除" in (b.label or "") for b in at.button)

    at = click_by_label(at, "确认删除")
    assert not at.exception
    assert current_step(at) == "history"
    assert database.get_concept(cid) is None
    assert database.get_qa_history(cid) == []
    assert any("还没有学习记录" in (i.value or "") for i in at.info)


def test_app_delete_from_list(configured_app) -> None:
    a = database.save_concept("机会成本", "原文A")
    b = database.save_concept("沉没成本", "原文B")
    database.update_concept(a, mastery="搞懂了")
    database.save_qa(b, "Q?", "A", True)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    assert current_step(at) == "history"
    # 按掌握度分组展示（搞懂了 / 学习中），没有「模糊」分组
    assert "✅ 搞懂了" in markdown_text(at)
    assert "📖 学习中" in markdown_text(at)
    assert "🔄 模糊" not in markdown_text(at)
    # 行内「查看」按钮切换详情选中
    at = at.button(key=f"view_{b}").click().run()
    assert not at.exception
    assert "history_view_id" in at.session_state
    assert at.session_state["history_view_id"] == b
    assert "沉没成本" in markdown_text(at)
    # 行内「✕」两步确认删除概念 A，删除后列表自动刷新
    at = at.button(key=f"xdel_{a}").click().run()
    assert not at.exception
    assert any("确认删除" in (btn.label or "") for btn in at.button)
    at = click_by_label(at, "确认删除")
    assert not at.exception
    assert current_step(at) == "history"
    assert database.get_concept(a) is None
    assert database.get_concept(b) is not None
    assert "机会成本" not in markdown_text(at)
    assert "📖 学习中" in markdown_text(at)
