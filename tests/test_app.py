"""Tests for the Streamlit UI using streamlit.testing.v1.AppTest."""

from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

from core import database
from core.client import DeepSeekNetworkError

APP = "app.py"


class FakeClient:
    """Scripted DeepSeek client reused for the UI flow."""

    def __init__(self, questions: list[dict], related=()) -> None:
        self.questions = list(questions)
        self.related = list(related)
        self.i = 0
        self.depth = 0
        # V0.3.0 Learning Loop v2 固定回复
        self.task_reply = {"task": "用一句话向朋友解释机会成本", "type": "summary", "difficulty": 2}
        self.analysis_reply = {
            "understanding_level": "relationship",
            "understood": ["理解了核心含义"],
            "uncertain": [],
            "misconceptions": [],
            "last_response_quality": "deep",
        }
        self.decider_reply = {
            "action": "hint",
            "reason": "",
            "content": "机会成本里的“成本”指的是什么？",
            "requires_user_response": True,
        }
        self.update_reply = {
            "understanding_level": "relationship",
            "understood": ["理解了核心含义"],
            "uncertain": [],
            "misconceptions": [],
            "last_response_quality": "deep",
            "next_best_action": "none",
        }
        self.offer_reply = {"offer": "要不要再挖一层？", "options": ["深入", "先到这里"]}

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
        if "讲得不够简单" in user:
            return "更简单的大白话：机会成本就是你为了A而放弃的B。"
        if "层深化" in user:
            self.depth += 1
            return f"深化问题{self.depth}"
        # ---- V0.3.0 Learning Loop v2 — prompt 分支 ----
        if "学习任务设计器" in user:
            return json.dumps(self.task_reply, ensure_ascii=False)
        if "学习者状态分析器" in user:
            return json.dumps(self.analysis_reply, ensure_ascii=False)
        if "最小干预决策器" in user:
            return json.dumps(self.decider_reply, ensure_ascii=False)
        if "学习状态更新器" in user:
            return json.dumps(self.update_reply, ensure_ascii=False)
        if "学习教练" in user:
            return json.dumps(self.offer_reply, ensure_ascii=False)
        raise AssertionError(f"unexpected prompt: {user[:40]}")


class FlakyClient(FakeClient):
    """FakeClient that can simulate AI timeout/network failures on demand."""

    def __init__(self, questions, *, fail_task: int = 0, fail_analyze: int = 0,
                 fail_decide: int = 0, fail_offer: int = 0) -> None:
        super().__init__(questions)
        self.fail_task = fail_task
        self.fail_analyze = fail_analyze
        self.fail_decide = fail_decide
        self.fail_offer = fail_offer
        self._task_calls = 0
        self._analyze_calls = 0
        self._decide_calls = 0
        self._offer_calls = 0

    def chat(self, messages, **kwargs) -> str:
        user = messages[1]["content"]
        if "学习任务设计器" in user:
            self._task_calls += 1
            if self._task_calls <= self.fail_task:
                raise DeepSeekNetworkError("Request timed out: 模拟超时")
        if "学习者状态分析器" in user:
            self._analyze_calls += 1
            if self._analyze_calls <= self.fail_analyze:
                raise DeepSeekNetworkError("Request timed out: 模拟超时")
        if "最小干预决策器" in user:
            self._decide_calls += 1
            if self._decide_calls <= self.fail_decide:
                raise DeepSeekNetworkError("Request timed out: 模拟超时")
        if "学习教练" in user:
            self._offer_calls += 1
            if self._offer_calls <= self.fail_offer:
                raise DeepSeekNetworkError("Request timed out: 模拟超时")
        return super().chat(messages, **kwargs)


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


@pytest.fixture
def legacy_flow(monkeypatch):
    """V0.3.0 — 固定旧流程（RECALLOS_NEW_FLOW=0），让旧四层追问测试保持原样。
    AppTest 会以新脚本进程执行 app.py，必须通过环境变量而不是模块属性来切换。"""
    monkeypatch.setenv("RECALLOS_NEW_FLOW", "0")
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


def test_app_full_flow(monkeypatch, configured_app, legacy_flow) -> None:
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
    # 表格按行展示掌握度标签与表头
    assert "✅ 搞懂了" in markdown_text(at)
    assert "概念名称" in markdown_text(at)
    assert "操作" in markdown_text(at)


def test_app_mode_toggle_on_home(configured_app, legacy_flow) -> None:
    """旧流程首页保留「有基础」切换（新流程首页不再显示）。"""
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert len(at.toggle) == 1
    assert "有基础" in at.toggle[0].label


def test_app_learning_flow_uses_dynamic_opening(monkeypatch, configured_app, legacy_flow) -> None:
    fake = FakeClient(make_questions(4))
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("机会成本")
    at.text_area[0].input("原文：选择意味着放弃")
    at = click_by_label(at, "开始")
    assert not at.exception
    assert current_step(at) == "learning"
    assert "开场问题" in markdown_text(at)


def test_app_explain_button(monkeypatch, configured_app, legacy_flow) -> None:
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient(make_questions(4)))
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("机会成本")
    at.text_area[0].input("原文：选择意味着放弃")
    at = click_by_label(at, "开始")
    at = click_by_label(at, "我不懂")
    assert not at.exception
    assert "我换个说法" in markdown_text(at)


def test_app_single_assistant_bubble_per_answer(monkeypatch, configured_app, legacy_flow) -> None:
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


def test_app_warmup_prepended_to_first_message(monkeypatch, configured_app, legacy_flow) -> None:
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


def test_app_delete_from_table(configured_app) -> None:
    a = database.save_concept("机会成本", "原文A")
    b = database.save_concept("沉没成本", "原文B")
    database.update_concept(a, mastery="搞懂了")
    database.save_qa(b, "Q?", "A", True)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    assert current_step(at) == "history"
    # 表格按行展示掌握度标签（搞懂了 / 学习中），无「模糊」行
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


def test_app_history_view_and_delete_buttons_via_callback(configured_app) -> None:
    """查看 / ✕ / 确认删除 全部走 on_click 回调：同组多概念时切换与删除即时生效。"""
    a = database.save_concept("机会成本", "原文A")
    b = database.save_concept("沉没成本", "原文B")
    database.update_concept(a, mastery="搞懂了")
    database.update_concept(b, mastery="搞懂了")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    assert current_step(at) == "history"
    # 点「查看」切换到 b，详情随之更新
    at = at.button(key=f"view_{b}").click().run()
    assert not at.exception
    assert at.session_state["history_view_id"] == b
    # 点「✕」弹出确认框，但数据暂未删除
    at = at.button(key=f"xdel_{a}").click().run()
    assert not at.exception
    assert any("确认删除" in (btn.label or "") for btn in at.button)
    assert database.get_concept(a) is not None
    # 点「确认删除」→ 回调删除并刷新列表，详情切回剩余概念
    at = click_by_label(at, "确认删除")
    assert not at.exception
    assert database.get_concept(a) is None
    assert database.get_concept(b) is not None
    assert "机会成本" not in markdown_text(at)
    assert "沉没成本" in markdown_text(at)


def test_app_delete_from_detail_page_returns_to_history(configured_app) -> None:
    """详情页顶部辅助删除：两步确认后回到历史页，被删概念已清除。"""
    cid_a = database.save_concept("机会成本", "原文A")
    cid_b = database.save_concept("沉没成本", "原文B")
    database.save_connection(cid_a, cid_b, "都关于选择")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    at = click_by_label(at, "去往")
    assert not at.exception
    assert current_step(at) == "concept_detail"
    shown = at.session_state["concept_detail_id"]
    # 详情页顶部有辅助删除入口，两步式确认
    assert any("删除这个概念" in (b.label or "") for b in at.button)
    at = click_by_label(at, "删除这个概念")
    assert not at.exception
    assert any("确认删除" in (b.label or "") for b in at.button)
    at = click_by_label(at, "确认删除")
    assert not at.exception
    assert current_step(at) == "history"
    assert database.get_concept(shown) is None


def test_app_history_view_expands_inline_and_collapses(configured_app) -> None:
    """「查看」在概念行下方行内展开详情；一次只展开一个；再次点击可收起。"""
    a = database.save_concept("机会成本", "原文A")
    b = database.save_concept("沉没成本", "原文B")
    database.update_concept(a, mastery="搞懂了")
    database.update_concept(b, mastery="搞懂了")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    assert current_step(at) == "history"
    # 首次进入默认展开第一个概念，行内只渲染一个展开区（一个「关闭」按钮）
    assert "history_view_id" in at.session_state
    assert at.session_state["history_view_id"] in (a, b)
    assert any("关闭" in (btn.label or "") for btn in at.button)
    # 点击「查看」切换展开到 b（一次只展开一个）
    at = at.button(key=f"view_{b}").click().run()
    assert not at.exception
    assert at.session_state["history_view_id"] == b
    assert any("关闭" in (btn.label or "") for btn in at.button)
    # 再次点击当前展开概念的「查看」→ 收起（不会被自动重新展开）
    at = at.button(key=f"view_{b}").click().run()
    assert not at.exception
    assert at.session_state["history_view_id"] is None
    assert not any("关闭" in (btn.label or "") for btn in at.button)
    # 可再点开另一个概念
    at = at.button(key=f"view_{a}").click().run()
    assert not at.exception
    assert at.session_state["history_view_id"] == a


# ---------------------------------------------------------------- V0.3.0 UI


def _start_new_flow(at: AppTest) -> AppTest:
    at.text_input[0].input("机会成本")
    at.text_area[0].input("原文：选择意味着放弃")
    return click_by_label(at, "开始")


def test_app_new_flow_home_drops_mode_toggle(configured_app) -> None:
    """新流程首页不再显示「有基础」模式切换。"""
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert not any("有基础" in (t.label or "") for t in at.toggle)
    assert any("概念名" in (ti.label or "") for ti in at.text_input)


def test_app_new_flow_reading_to_validation(monkeypatch, configured_app) -> None:
    fake = FakeClient([{"question": "Q", "correct": True, "feedback": "通过"}])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    assert not at.exception
    assert current_step(at) == "learning"
    session = at.session_state["session"]
    assert session.flow == "new"
    assert session.stage == "reading"
    assert any("阅读中" in (i.value or "") for i in at.info)
    assert any("先读原文" in m["text"] for m in at.session_state["messages"])

    at = click_by_label(at, "我读完了，开始验证")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"
    assert any("验证你的理解" in m["text"] for m in at.session_state["messages"])
    assert any("😵 我看不懂" in (b.label or "") for b in at.button)

    # 我看不懂 → 大白话解释，阶段不变
    at = click_by_label(at, "我看不懂")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"
    assert any("更简单的大白话" in m["text"] for m in at.session_state["messages"])


def test_app_new_flow_home_goal_selector(configured_app) -> None:
    """V0.3.1 — 首页可选学习目标；不选默认「理解概念」；选择会落库。"""
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert at.radio[0].value == "🧠 理解概念"

    at.radio[0].set_value("🛠 能实际应用").run()
    assert not at.exception
    at = _start_new_flow(at)
    assert not at.exception
    session = at.session_state["session"]
    assert session.learning_goal == "apply"
    assert database.get_concept(session.concept_id)["learning_goal"] == "apply"


def test_app_new_flow_reading_signal_buttons_and_stuck(
    monkeypatch, configured_app
) -> None:
    """V0.3.1 — 阅读阶段 🤔💡✓ 标记 + 卡住点输入，全部可选、不阻塞流程。"""
    fake = FakeClient([])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    assert at.session_state["session"].stage == "reading"

    # 点「🤔 没看懂」→ 记录信号并落库，页面出现卡住点输入
    at = at.button(key="v_rs_c_0").click().run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.reading_signals == [{"kind": "confused", "position": 0}]
    assert database.get_concept(session.concept_id)["signals"]
    assert "没看懂" in markdown_text(at)

    # 输入卡住点并保存
    at.text_input(key="v_stuck_text").input("边界不知道算不算").run()
    assert not at.exception
    at = click_by_label(at, "保存")
    assert not at.exception
    session = at.session_state["session"]
    assert session.stuck_points == ["边界不知道算不算"]

    # 再点「✓ 我懂了」也记录（混合信号不互相覆盖）
    at = at.button(key="v_rs_k_0").click().run()
    assert not at.exception
    session = at.session_state["session"]
    kinds = [s["kind"] for s in session.reading_signals]
    assert kinds == ["confused", "clear"]

    # 标记不影响流程：仍然可以正常开始验证
    at = click_by_label(at, "我读完了，开始验证")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"


def test_app_new_flow_intervention_feedback_ui(monkeypatch, configured_app) -> None:
    """V0.3.1 — 干预后 👍/🤔 反馈按钮可选；反馈写入 learner_state 并落库。"""
    fake = FakeClient([])
    fake.analysis_reply = {
        "understanding_level": "relationship",
        "understood": [],
        "uncertain": ["边界不清"],
        "misconceptions": [],
        "last_response_quality": "partial",
    }
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "我读完了，开始验证")
    at = at.chat_input[0].set_value("机会成本就是放弃的次优选择").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "intervention"
    assert session.feedback_pending() is True

    # 下一次干净 run 才渲染干预反馈按钮（处理回答那一 run 顶部还没到干预阶段）
    at = at.run()
    assert not at.exception
    assert any("👍 清楚多了" in (b.label or "") for b in at.button)

    # 提交「👍 清楚多了」反馈
    at = click_by_label(at, "👍 清楚多了")
    assert not at.exception
    session = at.session_state["session"]
    assert session.intervention_feedback_list == [
        {"action": "hint", "feedback": "clear"}
    ]
    assert (
        session.learner_state.intervention_history[-1]["feedback"] == "clear"
    )
    assert database.get_concept(session.concept_id)["intervention_feedback"]


def test_app_new_flow_validation_pass_goes_offer(monkeypatch, configured_app) -> None:
    fake = FakeClient([])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "我读完了，开始验证")
    at = at.chat_input[0].set_value("机会成本就是放弃的次优选择").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.validation_passed is True
    assert session.stage == "offer"
    # ✅ 通过气泡 + 自动生成的深入邀请气泡
    msgs = at.session_state["messages"]
    assert any("你已经理解核心概念" in m["text"] for m in msgs)
    assert any("要不要再挖一层" in m["text"] for m in msgs)
    # offer 阶段渲染出两个选择按钮
    assert any("我想再深入一层" in (b.label or "") for b in at.button)
    assert any("先到这里" in (b.label or "") for b in at.button)


def test_app_new_flow_validation_gap_enters_intervention(monkeypatch, configured_app) -> None:
    fake = FakeClient([])
    fake.analysis_reply = {
        "understanding_level": "relationship",
        "understood": [],
        "uncertain": ["边界不清"],
        "misconceptions": ["混淆了机会成本与沉没成本"],
        "last_response_quality": "partial",
    }
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "我读完了，开始验证")
    at = at.chat_input[0].set_value("机会成本就是选择").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.validation_passed is False
    assert session.stage == "intervention"
    assert any("机会成本里的“成本”指的是什么" in m["text"] for m in at.session_state["messages"])
    assert "最小干预" in markdown_text(at)


def test_app_new_flow_intervention_answer_stops_when_understand(monkeypatch, configured_app) -> None:
    fake = FakeClient([])
    fake.analysis_reply = {
        "understanding_level": "relationship",
        "understood": [],
        "uncertain": ["边界不清"],
        "misconceptions": ["混淆"],
        "last_response_quality": "partial",
    }
    fake.update_reply = {
        "understanding_level": "application",
        "understood": ["理解了核心与边界"],
        "uncertain": [],
        "misconceptions": [],
        "last_response_quality": "deep",
        "next_best_action": "none",
    }
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "我读完了，开始验证")
    at = at.chat_input[0].set_value("机会成本就是选择").run()  # 有缺口 → 干预
    assert not at.exception
    assert at.session_state["session"].stage == "intervention"
    # 回答干预 → updater 判断理解到位 → 无剩余干预 → 完成
    at = at.chat_input[0].set_value("成本是你放弃的那个最好的选择").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "complete"
    assert session.phase == "connections"
    assert any("这一步已经完成" in (s.value or "") for s in at.success)
    assert len(session.deeper_history) == 1
    assert session.deeper_history[0]["next_best_action"] == "none"


def test_app_new_flow_complete_goes_to_connections(monkeypatch, configured_app) -> None:
    fake = FakeClient([])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "我读完了，开始验证")
    at = at.chat_input[0].set_value("机会成本就是放弃的次优选择").run()  # 验证通过 → offer
    assert not at.exception
    # 用户主动选择「先到这里」→ 完成并进入复习队列
    at = click_by_label(at, "先到这里")
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "complete"
    assert session.phase == "connections"
    assert any("这一步已经完成" in (s.value or "") for s in at.success)

    at = click_by_label(at, "进入总结")
    assert not at.exception
    assert current_step(at) == "connections"
    assert "发现一些知识连接" in markdown_text(at)


def test_app_validation_timeout_retry_recovers(monkeypatch, configured_app) -> None:
    """验证分析 AI 超时：不转圈不崩，展示 ❌ 气泡，再次提交答案即可恢复。"""
    fake = FlakyClient([], fail_analyze=1)
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "我读完了，开始验证")
    at = at.chat_input[0].set_value("机会成本就是放弃的次优选择").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "validation"  # 失败不前进
    assert any("❌" in m["text"] for m in at.session_state["messages"] if m["text"])
    # 再次提交同一答案 → 分析成功 → 进入 offer
    at = at.chat_input[0].set_value("机会成本就是放弃的次优选择").run()
    assert not at.exception
    assert at.session_state["session"].stage == "offer"


def test_app_offer_timeout_shows_retry_and_recovers(monkeypatch, configured_app) -> None:
    """深入邀请 AI 超时：记录错误并展示重试按钮，重试成功后恢复 offer 阶段。"""
    fake = FlakyClient([], fail_offer=1)
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "我读完了，开始验证")
    at = at.chat_input[0].set_value("机会成本就是放弃的次优选择").run()  # 通过 → offer
    assert not at.exception
    assert at.session_state["session"].stage == "offer"
    assert ("v_ai_error" in at.session_state)
    assert any("重试" in (b.label or "") for b in at.button)
    # 点重试 → 第二次生成成功 → 显示深入/先到这里
    at = click_by_label(at, "重试")
    assert not at.exception
    assert "v_ai_error" not in at.session_state
    assert any("我想再深入一层" in (b.label or "") for b in at.button)


class JunkTaskClient(FakeClient):
    """FakeClient that returns non-JSON for the validation-task prompt."""

    def chat(self, messages, **kwargs) -> str:
        user = messages[1]["content"]
        if "学习任务设计器" in user:
            return "抱歉我这次没按格式输出"
        return super().chat(messages, **kwargs)


def test_app_start_validation_timeout_retryable(monkeypatch, configured_app) -> None:
    """读完后点「开始验证」时 AI 超时：不转圈不崩，留在阅读阶段，可重试恢复。"""
    fake = FlakyClient([], fail_task=1)
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "我读完了，开始验证")
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "reading"  # 失败不前进，留在阅读阶段可重试
    assert any("❌" in m["text"] for m in at.session_state["messages"] if m["text"])
    assert any("重试" in m["text"] for m in at.session_state["messages"] if m["text"])
    # 按钮还在 → 用户可以再点一次重试
    assert any("我读完了，开始验证" in (b.label or "") for b in at.button)

    # 重试 → 第二次成功 → 进入验证阶段
    at = click_by_label(at, "我读完了，开始验证")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"
    assert any("验证你的理解" in m["text"] for m in at.session_state["messages"])


def test_app_start_validation_malformed_reply_friendly_error(
    monkeypatch, configured_app
) -> None:
    """验证任务 AI 返回非 JSON：不抛未捕获异常，转为友好提示并可重试。"""
    monkeypatch.setattr(
        "core.session.DeepSeekClient", lambda: JunkTaskClient([])
    )
    at = AppTest.from_file(APP, default_timeout=30).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "我读完了，开始验证")
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "reading"  # 未崩溃、未前进，仍可重试
    assert any("格式不正确" in m["text"] for m in at.session_state["messages"] if m["text"])
    assert any("我读完了，开始验证" in (b.label or "") for b in at.button)


# ----------------------------------------------------- V0.3.0 session resume


def _seed_new_flow_concept(**updates) -> int:
    """Create a concept saved in the middle of the V0.3.0 new flow."""
    cid = database.save_concept("机会成本", "原文：选择意味着放弃")
    fields = {
        "stage": "validation",
        "validation_kind": "summary",
        "validation_difficulty": 2,
        "validation_task": "用一句话向朋友解释机会成本",
        "validation_target": "说出被放弃的次优选择",
        "validation_passed": False,
        "validation_attempts": 1,
        "validation_history": json.dumps(
            [
                {
                    "answer": "第一次回答",
                    "understanding_level": "relationship",
                    "last_response_quality": "partial",
                    "understood": [],
                    "uncertain": ["边界不清"],
                    "misconceptions": [],
                }
            ],
            ensure_ascii=False,
        ),
    }
    fields.update(updates)
    database.update_concept(cid, **fields)
    return cid


def test_app_continue_new_flow_resumes_validation(configured_app) -> None:
    cid = _seed_new_flow_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    at = click_by_label(at, "继续学习")
    assert not at.exception
    assert current_step(at) == "learning"

    session = at.session_state["session"]
    assert session.flow == "new"
    assert session.stage == "validation"
    assert session.validation_attempts == 1
    assert session.validation_task == "用一句话向朋友解释机会成本"
    assert session.learner_state.understanding_level == "relationship"

    # 重建的消息气泡：验证任务 + 上次回答 + 已分析气泡
    msgs = at.session_state["messages"]
    assert any("用一句话向朋友解释机会成本" in m["text"] for m in msgs)
    assert any(m["text"] == "第一次回答" for m in msgs)
    assert any("层级：relationship" in m["text"] for m in msgs)
    # UI 直接渲染验证任务（不依赖气泡）
    assert any("用一句话向朋友解释机会成本" in m.value for m in at.markdown)


def test_app_continue_new_flow_resumes_offer(monkeypatch, configured_app) -> None:
    """验证通过后中断：恢复后停在 offer 阶段，自动生成深入邀请等待用户选择。"""
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient([]))
    cid = _seed_new_flow_concept(
        stage="offer",
        validation_passed=True,
        validation_attempts=0,
        validation_history=json.dumps(
            [{
                "answer": "机会成本是我放弃的次优选择",
                "understanding_level": "relationship",
                "last_response_quality": "deep",
                "understood": ["核心"],
                "uncertain": [],
                "misconceptions": [],
            }],
            ensure_ascii=False,
        ),
    )
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    at = click_by_label(at, "继续学习")
    assert not at.exception
    assert current_step(at) == "learning"

    session = at.session_state["session"]
    assert session.flow == "new"
    assert session.stage == "offer"
    assert session.validation_passed is True
    # 重建气泡：通过通知 + 自动生成的深入邀请
    msgs = at.session_state["messages"]
    assert any("你已经理解核心概念" in m["text"] for m in msgs)
    assert any("要不要再挖一层" in m["text"] for m in msgs)
    assert any("我想再深入一层" in (b.label or "") for b in at.button)


def test_app_continue_new_flow_resumes_intervention(monkeypatch, configured_app) -> None:
    """干预阶段中断：恢复后自动为下一条干预决策并推到屏幕上。"""
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient([]))
    cid = _seed_new_flow_concept(
        stage="intervention",
        validation_passed=False,
        validation_attempts=1,
        validation_history=json.dumps(
            [{
                "answer": "第一次回答",
                "understanding_level": "relationship",
                "last_response_quality": "partial",
                "understood": [],
                "uncertain": ["边界不清"],
                "misconceptions": ["混淆"],
            }],
            ensure_ascii=False,
        ),
        deeper_answers=json.dumps(
            [{
                "question": "想一想成本指的是什么",
                "answer": "成本是你放弃的次优选择",
                "action": "hint",
                "understanding_level": "relationship",
                "last_response_quality": "partial",
                "understood": [],
                "uncertain": ["边界"],
                "misconceptions": [],
                "next_best_action": "hint",
            }],
            ensure_ascii=False,
        ),
        deeper_index=1,
    )
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    at = click_by_label(at, "继续学习")
    assert not at.exception
    assert current_step(at) == "learning"

    session = at.session_state["session"]
    assert session.flow == "new"
    assert session.stage == "intervention"
    assert session.current_intervention() is not None
    # 已答的那条干预还原在消息里，下一条由决策器自动生成
    msgs = at.session_state["messages"]
    assert any("想一想成本指的是什么" in m["text"] for m in msgs)
    assert any("机会成本里的“成本”指的是什么" in m["text"] for m in msgs)
    assert any("最小干预" in (i.value or "") for i in at.info)


# ------------------------------------------------- V0.3.0 history dual-flow


def _seed_new_flow_finished_concept() -> int:
    """Create a finished V0.3.0 new-flow concept with validation + deepening data."""
    cid = database.save_concept("机会成本", "原文：选择意味着放弃")
    database.update_concept(
        cid,
        mastery="搞懂了",
        user_definition="机会成本是放弃的次优选择",
        validation_type="definition",
        validation_task="用一句话向朋友解释机会成本",
        validation_target="说出被放弃的次优选择",
        validation_passed=True,
        validation_attempts=2,
        validation_history=json.dumps(
            [
                {
                    "answer": "第一次答错",
                    "understanding_level": "relationship",
                    "last_response_quality": "partial",
                    "understood": [],
                    "uncertain": ["边界"],
                    "misconceptions": [],
                },
                {
                    "answer": "机会成本是我放弃的次优选择",
                    "understanding_level": "application",
                    "last_response_quality": "deep",
                    "understood": ["核心"],
                    "uncertain": [],
                    "misconceptions": [],
                },
            ],
            ensure_ascii=False,
        ),
        deeper_questions=json.dumps(
            ["深化一：如果不考虑机会成本会怎样？", "深化二：机会成本和沉没成本什么关系？"],
            ensure_ascii=False,
        ),
        deeper_answers=json.dumps(
            [
                {"question": "深化一：如果不考虑机会成本会怎样？", "answer": "决策会失真"},
                {"question": "深化二：机会成本和沉没成本什么关系？", "answer": "两者都关于选择"},
            ],
            ensure_ascii=False,
        ),
        deeper_index=2,
    )
    return cid


def test_app_history_shows_new_flow_validation_and_deepening(configured_app) -> None:
    """有「我的理解」的新流程概念：历史页应显示验证任务/作答/结果与深化追问。"""
    _seed_new_flow_finished_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    assert current_step(at) == "history"

    text = markdown_text(at)
    assert "用一句话向朋友解释机会成本" in text       # 验证任务
    assert "第一次答错" in text                      # 第 1 次作答
    assert "机会成本是我放弃的次优选择" in text       # 第 2 次作答
    assert "✅ 通过" in text                          # 验证结果
    assert "深化一：如果不考虑机会成本会怎样？" in text
    assert "决策会失真" in text                       # 深化回答
    assert "我的回答：两者都关于选择" in text


def test_app_history_shows_new_flow_unfinished_without_definition(configured_app) -> None:
    """尚无 user_definition 的新流程概念：走 format_detail 也应显示验证/深化记录。"""
    cid = database.save_concept("机会成本", "原文：选择意味着放弃")
    database.update_concept(
        cid,
        validation_type="definition",
        validation_task="解释一下机会成本",
        validation_passed=False,
        validation_attempts=2,
        needs_relearning=True,
        validation_history=json.dumps(
            [{
                "answer": "答错",
                "understanding_level": "relationship",
                "last_response_quality": "partial",
                "understood": [],
                "uncertain": ["边界"],
                "misconceptions": [],
            }],
            ensure_ascii=False,
        ),
        deeper_questions=json.dumps(["不学机会成本会怎样？"], ensure_ascii=False),
        deeper_answers=json.dumps([]),
        deeper_index=1,
    )
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    assert current_step(at) == "history"

    text = markdown_text(at)
    assert "解释一下机会成本" in text
    assert "❌ 未通过（连续 3 次，需要重新学习）" in text
    assert "不学机会成本会怎样？" in text
    assert "（未回答）" in text


def test_app_history_legacy_qa_records_still_shown(configured_app) -> None:
    """旧流程概念（无 validation_type）：历史页追问记录仍来自 qa_records。"""
    cid = database.save_concept("机会成本", "原文")
    database.save_qa(cid, "它关注过去还是未来？", "未来", True)
    database.save_qa(cid, "和沉没成本的区别？", "一个看过去", False, hint_used=True)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "历史回顾")
    assert not at.exception
    assert current_step(at) == "history"

    text = markdown_text(at)
    assert "它关注过去还是未来？" in text
    assert "和沉没成本的区别？" in text
    assert "一个看过去" in text
    assert "（用过提示）" in text
