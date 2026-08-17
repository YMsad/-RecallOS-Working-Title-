"""Tests for the learning session flow (layers, hints, references, summary)."""

from __future__ import annotations

import json
from datetime import date, timedelta

import httpx
import pytest

from core import database
from core.client import DeepSeekClient
from core.config import Settings
from core.models import MASTERY_UNCLEAR, MASTERY_UNDERSTOOD
from core.session import (
    LearningSession,
    SessionError,
    restore_session,
    warmup_concept,
)

TEST_SETTINGS = Settings(
    deepseek_api_key="test-key",
    deepseek_base_url="https://api.deepseek.com/v1",
    deepseek_model="deepseek-chat",
    max_retries=1,
    retry_backoff=0.0,
    retry_jitter=0.0,
)


class ScriptedTransport:
    """Pops one response per request; records every request payload."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        assert self.responses, "no more scripted responses"
        return self.responses.pop(0)


def text_reply(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content}}]}
    )


def judge(is_correct: bool, feedback: str, hint=None) -> httpx.Response:
    body = {"is_correct": is_correct, "feedback": feedback, "hint": hint}
    return text_reply(json.dumps(body, ensure_ascii=False))


def connections_reply(*items) -> httpx.Response:
    return text_reply(json.dumps(list(items), ensure_ascii=False))


def summary_reply(breakthrough: str, hook: str) -> httpx.Response:
    return text_reply(
        json.dumps({"breakthrough": breakthrough, "tomorrow_hook": hook}, ensure_ascii=False)
    )


def validation_task_reply(task: str, kind: str = "summary", difficulty: int = 2) -> httpx.Response:
    return text_reply(
        json.dumps({"task": task, "type": kind, "difficulty": difficulty}, ensure_ascii=False)
    )


def analysis_reply(
    level="relationship",
    understood=(),
    uncertain=(),
    misconceptions=(),
    quality="partial",
) -> httpx.Response:
    body = {
        "understanding_level": level,
        "understood": list(understood),
        "uncertain": list(uncertain),
        "misconceptions": list(misconceptions),
        "last_response_quality": quality,
    }
    return text_reply(json.dumps(body, ensure_ascii=False))


def update_reply(
    level="relationship",
    understood=(),
    uncertain=(),
    misconceptions=(),
    quality="partial",
    next_best_action="none",
) -> httpx.Response:
    body = {
        "understanding_level": level,
        "understood": list(understood),
        "uncertain": list(uncertain),
        "misconceptions": list(misconceptions),
        "last_response_quality": quality,
        "next_best_action": next_best_action,
    }
    return text_reply(json.dumps(body, ensure_ascii=False))


def intervention_reply(
    action="hint", content="想一想成本指的是什么", requires_user_response=True, reason=None
) -> httpx.Response:
    body = {
        "action": action,
        "content": content,
        "requires_user_response": requires_user_response,
        "reason": reason,
    }
    return text_reply(json.dumps(body, ensure_ascii=False))


def offer_reply(offer: str = "要不要再挖一层？") -> httpx.Response:
    return text_reply(
        json.dumps({"offer": offer, "options": ["深入", "先到这里"]}, ensure_ascii=False)
    )


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    database.configure(tmp_path / "test.db")
    yield


def make_session(*responses: httpx.Response, title="机会成本",
                 source="原文：选择意味着放弃", max_fail=3, mode="beginner"):
    transport = ScriptedTransport(list(responses))
    client = DeepSeekClient(
        settings=TEST_SETTINGS, transport=httpx.MockTransport(transport.handler)
    )
    session = LearningSession(
        title, source, client=client, max_consecutive_failures=max_fail, mode=mode
    )
    return session, transport


def test_start_saves_concept_and_returns_question() -> None:
    session, _ = make_session(text_reply("Q1：机会成本的核心是什么？"))
    question = session.start()
    assert question == "Q1：机会成本的核心是什么？"
    assert session.layer == 1
    assert session.phase == "learning"
    row = database.get_concept(session.concept_id)
    assert row["title"] == "机会成本"
    assert row["mastery"] == "学习中"


def test_begin_saves_concept_and_enters_reading_stage() -> None:
    """V0.3.0 — 新流程 begin()：只存概念、不生成开场问题，进入阅读阶段。"""
    session, transport = make_session()
    cid = session.begin()
    assert cid is not None
    assert transport.requests == []  # 不调用任何 AI，不消耗脚本
    assert session.stage == "reading"
    assert session.flow == "legacy"
    assert session.phase == "learning"
    assert database.get_concept(cid)["title"] == "机会成本"
    # 幂等：重复调用返回同一个 concept_id
    assert session.begin() == cid


def test_full_flow_all_correct() -> None:
    database.save_concept("沉没成本", "已学过的概念")
    session, transport = make_session(
        text_reply("Q1"),
        judge(True, "对"),
        text_reply("Q2"),
        judge(True, "对"),
        text_reply("Q3"),
        judge(True, "对"),
        text_reply("Q4"),
        judge(True, "对"),
        connections_reply(
            {"concept_title": "沉没成本", "relation_text": "都关于选择，一个看未来一个看过去"}
        ),
        summary_reply("我终于搞懂了机会成本", "边际效用和机会成本有什么关系？"),
    )
    session.start()
    for _ in range(4):
        result = session.submit_answer("我的回答")
        assert result["correct"] is True

    assert session.phase == "connections"
    assert session.next_question() is None

    # Layer 4 question was seeded with previously learned concepts
    layer4_payload = next(
        p for p in transport.requests if "第四层追问" in p["messages"][1]["content"]
    )
    assert "沉没成本" in layer4_payload["messages"][1]["content"]

    conns = session.get_connections()
    assert len(conns) == 1
    assert conns[0].concept_title == "沉没成本"
    assert len(database.get_connections(session.concept_id)) == 1

    summary = session.finish(user_definition="机会成本是放弃的价值")
    assert summary.breakthrough == "我终于搞懂了机会成本"
    assert session.phase == "finished"

    today = database.get_today_summary()
    assert today["breakthrough_text"] == "我终于搞懂了机会成本"
    assert today["tomorrow_hook"] == "边际效用和机会成本有什么关系？"
    row = database.get_concept(session.concept_id)
    assert row["mastery"] == MASTERY_UNDERSTOOD
    assert row["user_definition"] == "机会成本是放弃的价值"


def test_wrong_answer_returns_hint_and_stays() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        judge(False, "再想想", "想一想你放弃了什么"),
        text_reply("更简单的问题：你买奶茶时放弃了什么？"),
    )
    session.start()
    result = session.submit_answer("错误回答")
    assert result["correct"] is False
    assert result["hint"] == "想一想你放弃了什么"
    assert result["reference"] is None
    assert result["is_done"] is False
    assert result["simplified"] is True
    assert session.layer == 1
    # 降维追问：不再是原问题，而是更简单的问题
    assert session.next_question() == "更简单的问题：你买奶茶时放弃了什么？"
    history = database.get_qa_history(session.concept_id)
    assert len(history) == 1
    assert history[0]["hint_used"] == 1
    assert history[0]["is_correct"] == 0


def test_three_failures_escalate_simplify_angle_reference() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        judge(False, "再想想", "h1"),
        text_reply("简化后的问题"),
        judge(False, "再想想", "h2"),
        text_reply("换个角度的问题"),
        judge(False, "再想想", "h3"),
        text_reply("参考解释：机会成本是你放弃的次优价值"),
        text_reply("Q2"),
    )
    session.start()
    r1 = session.submit_answer("A1")
    assert r1["simplified"] is True
    assert r1["angle_shift"] is False
    assert r1["reference"] is None
    assert session.next_question() == "简化后的问题"

    r2 = session.submit_answer("A2")
    assert r2["angle_shift"] is True
    assert r2["reference"] is None
    assert session.next_question() == "换个角度的问题"

    r3 = session.submit_answer("A3")
    assert r3["reference"] == "参考解释：机会成本是你放弃的次优价值"
    assert r3["mastery"] == MASTERY_UNCLEAR
    assert r3["is_done"] is False
    assert session.layer == 2
    assert session.next_question() == "Q2"
    assert database.get_concept(session.concept_id)["mastery"] == MASTERY_UNCLEAR


def test_uncertain_flow_finishes_with_mastery_uncertain() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        judge(False, "再想想", "h"),
        text_reply("简化"),
        judge(False, "再想想", "h"),
        text_reply("换角度"),
        judge(False, "再想想", "h"),
        text_reply("参考解释"),
        text_reply("Q2"),
        judge(True, "对"),
        text_reply("Q3"),
        judge(True, "对"),
        text_reply("Q4"),
        judge(True, "对"),
        connections_reply({"concept_title": "沉没成本", "relation_text": "R"}),
        summary_reply("有收获", "明天接着想"),
    )
    session.start()
    session.submit_answer("错")
    session.submit_answer("错")
    session.submit_answer("错")
    session.submit_answer("对")
    session.submit_answer("对")
    session.submit_answer("对")
    assert session.phase == "connections"
    session.get_connections()
    session.finish(user_definition="有点明白了")
    assert database.get_concept(session.concept_id)["mastery"] == MASTERY_UNCLEAR


def test_explain_mode_returns_plain_text_and_resets_failures() -> None:
    session, transport = make_session(
        text_reply("Q1"),
        judge(False, "再想想", "h"),
        text_reply("简化"),
        judge(False, "再想想", "h"),
        text_reply("换个角度的问题"),
        text_reply("大白话解释：机会成本就是你放弃的那个次优选择"),
    )
    session.start()
    session.submit_answer("错误")
    session.submit_answer("再错")
    assert session.consecutive_failures == 2
    explanation = session.explain()
    assert explanation == "大白话解释：机会成本就是你放弃的那个次优选择"
    assert session.explain_used is True
    assert session.consecutive_failures == 0
    explain_payload = next(
        p for p in transport.requests if "我不懂" in p["messages"][1]["content"]
    )
    assert "大白话" in explain_payload["messages"][1]["content"]


def test_ask_for_angle_switch_explicit() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        text_reply("换个角度的问题"),
    )
    session.start()
    question = session.ask_for_angle_switch()
    assert question == "换个角度的问题"
    assert session.consecutive_failures == 0
    assert session.next_question() == "换个角度的问题"


def test_opening_question_is_tailored_to_level_and_interest() -> None:
    session, transport = make_session(text_reply("开场问题"))
    session.start()
    opening_payload = transport.requests[0]["messages"][1]["content"]
    assert "开场第一个问题" in opening_payload
    assert "零基础" in opening_payload or "基础" in opening_payload
    assert "兴趣" in opening_payload


def test_warmup_returns_plain_text_for_beginner_mode() -> None:
    session, transport = make_session(text_reply("机会成本就是你为了得到A而放弃的B。"))
    warmup = session.warmup()
    assert warmup == "机会成本就是你为了得到A而放弃的B。"
    warmup_payload = transport.requests[0]["messages"][1]["content"]
    assert "预热" in warmup_payload
    assert "用一句话说" in warmup_payload


def test_warmup_skipped_for_advanced_mode() -> None:
    session, _ = make_session()
    session.mode = "advanced"
    assert session.warmup() == ""
    assert session._current_question is None


def test_warmup_concept_standalone_function() -> None:
    transport = ScriptedTransport(
        [text_reply("机会成本就是你为了得到A而放弃的B。")]
    )
    client = DeepSeekClient(
        settings=TEST_SETTINGS, transport=httpx.MockTransport(transport.handler)
    )
    warmup = warmup_concept("机会成本", "原文：选择意味着放弃", client=client)
    assert warmup == "机会成本就是你为了得到A而放弃的B。"
    payload = transport.requests[0]["messages"][1]["content"]
    assert "预热" in payload
    assert "用一句话说" in payload


def test_cognitive_contrast_used_in_beginner_layers() -> None:
    session, transport = make_session(
        text_reply("Q1"),
        judge(True, "对"),
        text_reply("Q2"),
        judge(True, "对"),
        text_reply("Q3"),
        judge(True, "对"),
        text_reply("Q4"),
        judge(True, "对"),
    )
    session.start()
    for _ in range(3):
        session.submit_answer("答")
    layer2_payload = next(
        p for p in transport.requests if "第二层追问" in p["messages"][1]["content"]
    )
    assert "认知反差" in layer2_payload["messages"][1]["content"]
    layer3_payload = next(
        p for p in transport.requests if "第三层追问" in p["messages"][1]["content"]
    )
    assert "认知反差" in layer3_payload["messages"][1]["content"]
    layer4_payload = next(
        p for p in transport.requests if "第四层追问" in p["messages"][1]["content"]
    )
    assert "认知反差" not in layer4_payload["messages"][1]["content"]


def test_cognitive_contrast_not_used_in_advanced_mode() -> None:
    session, transport = make_session(
        text_reply("Q1"),
        judge(True, "对"),
        text_reply("Q2"),
    )
    session.mode = "advanced"
    session.start()
    session.submit_answer("答")
    layer2_payload = next(
        p for p in transport.requests if "第二层追问" in p["messages"][1]["content"]
    )
    assert "认知反差" not in layer2_payload["messages"][1]["content"]


def test_out_of_order_calls_raise() -> None:
    session, _ = make_session(text_reply("Q1"))
    with pytest.raises(SessionError):
        session.submit_answer("x")
    with pytest.raises(SessionError):
        session.get_connections()
    with pytest.raises(SessionError):
        session.finish()
    session.start()
    with pytest.raises(SessionError):
        session.get_connections()  # still learning


def test_connections_skip_unmatched_titles() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        judge(True, "对"),
        text_reply("Q2"),
        judge(True, "对"),
        text_reply("Q3"),
        judge(True, "对"),
        text_reply("Q4"),
        judge(True, "对"),
        connections_reply({"concept_title": "不存在的概念", "relation_text": "R"}),
    )
    session.start()
    for _ in range(4):
        session.submit_answer("x")
    conns = session.get_connections()
    assert len(conns) == 1
    assert database.get_all_connections() == []  # nothing persisted


# ------------------------------------------------------- V0.2.1 thinking models


def test_route_layer4_uses_analogy() -> None:
    session, _ = make_session(text_reply("Q1"))
    assert session._route_model(4) == "analogy"


def test_route_beginner_uses_scenario_and_golden_circle() -> None:
    session, _ = make_session(text_reply("Q1"))
    assert session._route_model(1) == "scenario"
    assert session._route_model(2) == "scenario"
    assert session._route_model(3) == "golden_circle"


def test_route_advanced_streak_layer3_uses_golden_circle() -> None:
    session, _ = make_session(text_reply("Q1"), mode="advanced")
    assert session._route_model(1) == "first_principles"
    assert session._route_model(2) == "first_principles"
    assert session._route_model(3) == "golden_circle"


def test_route_advanced_streak_uses_first_principles() -> None:
    session, _ = make_session(text_reply("Q1"), mode="advanced")
    assert session._route_model(1) == "first_principles"


def test_route_advanced_after_failure_uses_golden_circle() -> None:
    session, _ = make_session(text_reply("Q1"), mode="advanced")
    session.consecutive_failures = 1
    assert session._route_model(1) == "golden_circle"


def test_route_advanced_uncertain_uses_golden_circle() -> None:
    session, _ = make_session(text_reply("Q1"), mode="advanced")
    session.marked_uncertain = True
    assert session._route_model(2) == "golden_circle"


def test_question_payload_includes_routed_model() -> None:
    session, transport = make_session(
        text_reply("Q1"), judge(True, "对"), text_reply("Q2"), mode="advanced"
    )
    session.start()
    session.submit_answer("答")
    payload = next(
        p for p in transport.requests if "第二层追问" in p["messages"][1]["content"]
    )
    assert "思维模型【第一性原理】" in payload["messages"][1]["content"]


# ------------------------------------------------------- V0.2.3 review queue


def test_learning_complete_queues_review_without_finish() -> None:
    """学习完成后（即使不进入总结），概念也必须加入复习队列。"""
    session, _ = make_session(
        text_reply("Q1"),
        judge(True, "对"),
        text_reply("Q2"),
        judge(True, "对"),
        text_reply("Q3"),
        judge(True, "对"),
        text_reply("Q4"),
        judge(True, "对"),
    )
    session.start()
    for _ in range(4):
        session.submit_answer("答")
    assert session.phase == "connections"
    row = database.get_concept(session.concept_id)
    expected = (date.today() + timedelta(days=1)).isoformat()
    assert row["next_review_date"] == expected


# ------------------------------------------------------- V0.3.0 validation stage
# Learning Loop v2：验证任务 → 学习者状态分析 → 最小干预 → 动态结束


def test_start_validation_designs_task_and_sets_stage() -> None:
    session, transport = make_session(
        validation_task_reply("用一句话向朋友解释机会成本", "summary", 2),
    )
    cid = session.begin()
    task_text = session.start_validation()
    assert task_text == "用一句话向朋友解释机会成本"
    assert session.validation_kind == "summary"
    assert session.validation_difficulty == 2
    assert session.stage == "validation"
    assert session.validation_passed is False
    assert session.needs_relearning is False
    payload = transport.requests[0]["messages"][1]["content"]
    assert "验证" in payload
    assert '"task"' in payload
    # learner state 被重置
    assert session.learner_state.understanding_level == "surface"
    assert session.current_intervention() is None
    assert session._offer is None


def test_start_validation_malformed_reply_raises_friendly_session_error() -> None:
    """AI 返回非 JSON：转成可重试的 SessionError，而不是未捕获的校验异常。"""
    session, _ = make_session(text_reply("抱歉，我这次没按格式输出"))
    session.begin()
    with pytest.raises(SessionError, match="格式不正确"):
        session.start_validation()
    assert session.stage == "reading"  # 失败不前进


def test_submit_validation_no_gap_completes() -> None:
    """V0.3.1 — 验证过即完：无理解缺口直接完成，不再进入 offer。"""
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", understood=["理解了核心含义"], quality="deep"),
    )
    cid = session.begin()
    session.start_validation()
    result = session.submit_validation("机会成本就是我必须放弃的那个次优选择")
    assert result["stage"] == "complete"
    assert "你的理解已经到位" in result["final_note"]
    assert session.validation_passed is True
    assert session.validation_attempts == 0
    assert session.stage == "complete"
    assert session.phase == "connections"
    assert len(session.validation_history) == 1
    assert session.learner_state.has_gap() is False
    # 完成即加入复习队列
    row = database.get_concept(cid)
    assert row["next_review_date"] == (date.today() + timedelta(days=1)).isoformat()


def test_submit_validation_gap_moves_to_intervention() -> None:
    session, transport = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", uncertain=["边界不清"], misconceptions=["混淆了机会成本与沉没成本"]),
        intervention_reply("counterexample", "如果放弃三个选择，机会成本是三个加起来吗？"),
    )
    cid = session.begin()
    session.start_validation()
    result = session.submit_validation("回答")
    assert result["stage"] == "intervention"
    assert "如果放弃三个选择" in result["bubble"]
    assert session.stage == "intervention"
    assert session.validation_passed is False
    assert session.current_intervention()["action"] == "counterexample"
    assert len(session.validation_history) == 1
    # 决策器/分析器 prompt 都体现了学习者状态
    payloads = [p["messages"][1]["content"] for p in transport.requests]
    assert any("学习者状态分析器" in p for p in payloads)
    assert any("最小干预决策器" in p for p in payloads)


def test_submit_validation_closing_note_finishes() -> None:
    """决策器返回 action=none（收尾内容、无需用户作答）→ 直接完成整个流程。"""
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", uncertain=["边界不清"]),
        intervention_reply("none", "你已经理解核心，边界问题已不值得继续追问。", requires_user_response=False),
    )
    cid = session.begin()
    session.start_validation()
    result = session.submit_validation("回答")
    assert result["stage"] == "complete"
    assert "你已经理解核心" in result["final_note"]
    assert session.stage == "complete"
    assert session.phase == "connections"
    assert session.current_intervention() is None
    row = database.get_concept(cid)
    assert row["next_review_date"] == (date.today() + timedelta(days=1)).isoformat()


def test_submit_validation_requires_validation_stage() -> None:
    session, _ = make_session()
    session.begin()
    with pytest.raises(SessionError):
        session.submit_validation("x")


def test_ask_simplify_returns_plain_text_and_keeps_stage() -> None:
    session, transport = make_session(
        text_reply("Q1"),
        text_reply("更简单的大白话：机会成本就是你为了A而放弃的B"),
    )
    session.start()
    session.stage = "validation"
    simple = session.ask_simplify()
    assert simple == "更简单的大白话：机会成本就是你为了A而放弃的B"
    assert session.stage == "validation"
    payload = transport.requests[1]["messages"][1]["content"]
    assert "讲得不够简单" in payload


# ------------------------------------------------------ V0.3.0 deepen offer loop


def test_offer_deepening_generates_offer() -> None:
    """V0.3.1 — offer API 仍保留（仅兼容遗留数据）：须先处于 offer 阶段。"""
    session, transport = make_session(
        validation_task_reply("任务"),
        offer_reply("你已经抓住核心，要不要再挖一层？"),
    )
    session.begin()
    session.start_validation()
    session.stage = "offer"  # 主流程不再进入 offer；此处显式置入以测遗留 API
    result = session.offer_deepening()
    assert result["offer"] == "你已经抓住核心，要不要再挖一层？"
    assert result["options"] == ["深入", "先到这里"]
    assert session.stage == "offer"
    payload = transport.requests[1]["messages"][1]["content"]
    assert "继续深入" in payload


def test_offer_deepening_requires_offer_stage() -> None:
    session, _ = make_session()
    session.begin()
    with pytest.raises(SessionError):
        session.offer_deepening()


def test_choose_deepening_stop_finishes() -> None:
    """V0.3.1 — 遗留 offer 阶段下「先到这里」仍然完成整个流程。"""
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", understood=["核心"], quality="deep"),
    )
    cid = session.begin()
    session.start_validation()
    session.stage = "offer"
    result = session.choose_deepening(False)
    assert result["stage"] == "complete"
    assert session.stage == "complete"
    assert session.phase == "connections"
    row = database.get_concept(cid)
    assert row["next_review_date"] == (date.today() + timedelta(days=1)).isoformat()


def test_choose_deepening_requires_offer_stage() -> None:
    session, _ = make_session()
    session.begin()
    with pytest.raises(SessionError):
        session.choose_deepening(True)


def test_choose_deepening_go_starts_intervention_loop() -> None:
    """V0.3.1 — 遗留 offer 阶段下选择深入仍可进入干预循环。"""
    session, _ = make_session(
        validation_task_reply("任务"),
        offer_reply("要不要再挖一层？"),
        intervention_reply("example", "想想你周末放弃游戏时间的选择"),
        update_reply("application", understood=["核心", "会应用"], next_best_action="none"),
    )
    cid = session.begin()
    session.start_validation()
    session.stage = "offer"
    session.offer_deepening()
    result = session.choose_deepening(True)
    assert result["stage"] == "intervention"
    assert "想想你周末" in result["bubble"]
    assert session.stage == "intervention"
    assert session.current_intervention()["action"] == "example"

    # 用户回答 → updater 判断巨大进步（无缺口 + next_best_action=none）→ 再决策 → 无剩余干预 → 完成
    result = session.submit_intervention_answer("周末我会权衡取舍")
    assert result["stage"] == "complete"
    assert session.stage == "complete"
    assert session.phase == "connections"
    assert len(session.deeper_history) == 1
    assert session.deeper_history[0]["question"].startswith("想想你周末")
    assert session.deeper_history[0]["answer"] == "周末我会权衡取舍"
    assert session.deeper_history[0]["understanding_level"] == "application"
    assert database.get_concept(cid)["mastery"] is not None


def test_submit_intervention_answer_requires_active_intervention() -> None:
    session, _ = make_session()
    session.begin()
    with pytest.raises(SessionError):
        session.submit_intervention_answer("x")


def test_next_intervention_continues_after_restore() -> None:
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", uncertain=["边界"], misconceptions=["混淆"]),
        intervention_reply("counterexample", "如果放弃三个选择，机会成本是三个加起来吗？"),
        intervention_reply("question", "机会成本里的“成本”指的是什么？"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("回答")
    assert session.stage == "intervention"

    restored = restore_session(cid, client=session.client)
    assert restored.flow == "new"
    assert restored.stage == "intervention"
    assert restored.current_intervention() is None  # 未回答的干预不落库

    result = restored.next_intervention()
    assert result["stage"] == "intervention"
    assert "机会成本里" in result["bubble"]
    assert restored.stage == "intervention"


# ---------------------------------------------------- V0.3.0 learner state unit


def test_learner_state_level_never_regresses() -> None:
    from core.learner_state import LearnerState

    state = LearnerState()
    state.update_from_analysis(
        {
            "understanding_level": "relationship",
            "understood": ["x"],
            "uncertain": [],
            "misconceptions": [],
            "last_response_quality": "deep",
        }
    )
    assert state.understanding_level == "relationship"
    # AI 判断回退到低层级时，保留历史最高层级
    state.update_from_analysis(
        {
            "understanding_level": "surface",
            "understood": ["x"],
            "uncertain": ["y"],
            "misconceptions": [],
            "last_response_quality": "shallow",
        }
    )
    assert state.understanding_level == "relationship"
    assert state.has_gap() is True


def test_learner_state_should_stop_logic() -> None:
    from core.learner_state import LearnerState

    assert LearnerState(understanding_level="relationship", next_best_action="none").should_stop() is True
    assert LearnerState(
        understanding_level="relationship", uncertain=["边界"], next_best_action="none"
    ).should_stop() is False
    # 无缺口但 AI 仍认为值得行动 → 不停
    assert LearnerState(
        understanding_level="relationship", next_best_action="hint"
    ).should_stop() is False


# -------------------------------------------------------- V0.3.0 deeper questions


def test_next_deeper_question_runs_full_sequence() -> None:
    session, transport = make_session(
        text_reply("Q1"),
        text_reply("再验证问题"),
        text_reply("联系问题"),
        text_reply("反事实问题"),
        text_reply("行动问题"),
        text_reply("第一性原理问题"),
    )
    session.start()

    questions = []
    while True:
        q = session.next_deeper_question()
        if q is None:
            break
        questions.append(q)

    assert questions == [
        "再验证问题",
        "联系问题",
        "反事实问题",
        "行动问题",
        "第一性原理问题",
    ]
    assert session.current_deeper_index == 5
    assert len(session.deeper_questions) == 5
    assert session.next_deeper_question() is None
    # 深化追问全部问完 → 新流程学习完成并加入复习队列
    assert session.stage == "complete"
    assert session.phase == "connections"

    deeper_payloads = [
        p["messages"][1]["content"]
        for p in transport.requests
        if "层深化" in p["messages"][1]["content"]
    ]
    assert len(deeper_payloads) == 5
    assert "再验证层深化" in deeper_payloads[0]
    assert "联系层深化" in deeper_payloads[1]
    assert "反事实层深化" in deeper_payloads[2]
    assert "行动层深化" in deeper_payloads[3]
    assert "第一性原理层深化" in deeper_payloads[4]


def test_submit_deeper_answer_records_exchange() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        text_reply("再验证问题"),
    )
    session.start()
    q = session.next_deeper_question()
    assert q == "再验证问题"
    result = session.submit_deeper_answer("我觉得机会成本就是放弃的次优")
    assert result["recorded"] is True
    assert result["question"] == "再验证问题"
    assert len(session.deeper_history) == 1
    assert session.deeper_history[0]["answer"] == "我觉得机会成本就是放弃的次优"


def test_submit_deeper_answer_requires_open_question() -> None:
    session, _ = make_session(text_reply("Q1"))
    session.start()
    with pytest.raises(SessionError):
        session.submit_deeper_answer("x")


# ------------------------------------------------------- V0.3.0 session restore


def test_restore_legacy_flow_rebuilds_qa_history() -> None:
    """旧流程启动的概念（无 stage/validation_type 标记）恢复后仍是旧流程。"""
    session, _ = make_session(text_reply("Q1"), judge(True, "对"), text_reply("Q2"))
    session.start()
    session.submit_answer("答")
    assert database.get_concept(session.concept_id)["stage"] is None

    restored = restore_session(session.concept_id)
    assert restored.flow == "legacy"
    assert restored.concept_id == session.concept_id
    assert len(restored.qa_history) == 1
    assert restored.layer == 1
    assert restored._current_question == "Q1"


def test_restore_new_flow_reading_stage() -> None:
    session, _ = make_session()
    cid = session.begin()
    assert database.get_concept(cid)["stage"] == "reading"

    restored = restore_session(cid)
    assert restored.flow == "new"
    assert restored.stage == "reading"
    assert restored.concept_id == cid
    assert restored.title == "机会成本"


def test_restore_new_flow_validation_preserves_state() -> None:
    session, _ = make_session(
        validation_task_reply("任务", "summary", 2),
        analysis_reply("relationship", uncertain=["边界"]),
        intervention_reply("hint", "想一想成本指的是什么"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("回答1")
    assert session.stage == "intervention"

    restored = restore_session(cid)
    assert restored.flow == "new"
    assert restored.stage == "intervention"
    assert restored.validation_task == "任务"
    assert restored.validation_kind == "summary"
    assert restored.validation_difficulty == 2
    assert restored.validation_attempts == 1
    assert restored.current_intervention() is None  # 未回答的干预不落库
    assert len(restored.validation_history) == 1
    assert restored.validation_history[0]["understanding_level"] == "relationship"
    assert restored.learner_state.understanding_level == "relationship"


def test_restore_new_flow_offer_stage() -> None:
    """V0.3.1 — 遗留 offer 阶段数据（老版本生成的）仍能恢复，交给 UI 自动收尾。"""
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", understood=["核心"], quality="deep"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("回答")
    assert session.stage == "complete"  # 主流程不再生成 offer
    # 模拟老数据：把阶段手工改回 offer 并落库
    session.stage = "offer"
    session._persist_new_flow()

    restored = restore_session(cid)
    assert restored.flow == "new"
    assert restored.stage == "offer"
    assert restored.validation_passed is True
    assert restored.validation_kind == "summary"
    assert restored.learner_state.understanding_level == "relationship"


def test_restore_new_flow_complete_sets_connections_phase() -> None:
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", understood=["核心"], quality="deep"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("回答")
    assert session.stage == "complete"
    assert session.phase == "connections"

    restored = restore_session(cid)
    assert restored.flow == "new"
    assert restored.stage == "complete"
    assert restored.phase == "connections"
    assert restored.learner_state.understanding_level == "relationship"
    assert restored.current_intervention() is None


# ------------------------------------------------------ V0.3.1 auto summary


def test_finish_auto_generates_summary_from_validation() -> None:
    """V0.3.1 — 完成即自动生成总结：用最新验证回答做素材，落库并更新掌握度。"""
    session, transport = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", understood=["理解了核心含义"], quality="deep"),
        summary_reply("我终于搞懂了机会成本", "明天想一个反例"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("机会成本就是我放弃的那个最好的选择")
    assert session.stage == "complete"

    summary = session.finish_auto()
    assert summary.breakthrough == "我终于搞懂了机会成本"
    assert summary.tomorrow_hook == "明天想一个反例"
    assert session.phase == "finished"

    row = database.get_concept(cid)
    assert row["mastery"] == "搞懂了"
    assert row["user_definition"] == "机会成本就是我放弃的那个最好的选择"
    today = database.get_today_summary()
    assert today is not None
    assert today["breakthrough_text"] == "我终于搞懂了机会成本"
    assert today["tomorrow_hook"] == "明天想一个反例"
    payload = transport.requests[2]["messages"][1]["content"]
    assert "机会成本就是我放弃的那个最好的选择" in payload


def test_finish_auto_requires_complete_stage() -> None:
    session, _ = make_session()
    session.begin()
    with pytest.raises(SessionError, match="complete"):
        session.finish_auto()


# ------------------------------------------------------ V0.3.1 user signals


def test_learning_goal_stored_and_restored() -> None:
    """学习目标可选；不选/非法取值回退 understand，且随会话恢复。"""
    session = LearningSession("机会成本", "原文", learning_goal="apply")
    assert session.learning_goal == "apply"
    cid = session.begin()

    row = database.get_concept(cid)
    assert row["learning_goal"] == "apply"

    restored = restore_session(cid)
    assert restored.learning_goal == "apply"
    assert restored.learner_state.learning_goal == "apply"

    fallback = LearningSession("x", "y", learning_goal="bogus")
    assert fallback.learning_goal == "understand"
    default = LearningSession("x", "y")
    assert default.learning_goal == "understand"


def test_reading_signals_persist_and_restore() -> None:
    """阅读阶段的 🤔💡✓ 标记与卡住点：记录、落库、恢复后可用。"""
    session = LearningSession("机会成本", "段一\n\n段二\n\n段三")
    cid = session.begin()
    session.record_reading_signal("confused", 0)
    session.record_reading_signal("clear", 2)
    session.record_stuck_point("机会成本到底只算最优那一个还是都算")

    assert session.reading_signals == [
        {"kind": "confused", "position": 0},
        {"kind": "clear", "position": 2},
    ]
    # V0.3.0 patch 1 — 标「没看懂」自动进 stuck_points / learner_state.uncertain
    derived = "阅读原文第 1 段时用户标了「没看懂」"
    assert session.stuck_points == [derived, "机会成本到底只算最优那一个还是都算"]
    assert session.learner_state.uncertain == [
        derived,
        "机会成本到底只算最优那一个还是都算",
    ]
    assert session.learner_state.has_gap() is True

    restored = restore_session(cid)
    assert restored.reading_signals == [
        {"kind": "confused", "position": 0},
        {"kind": "clear", "position": 2},
    ]
    assert restored.stuck_points == [derived, "机会成本到底只算最优那一个还是都算"]
    assert "机会成本到底只算最优那一个还是都算" in restored.learner_state.uncertain
    assert derived in restored.learner_state.uncertain


def test_confused_signal_alone_closes_gap() -> None:
    """V0.3.0 patch 1 — 只点「🤔 没看懂」不填文字，也算一个理解缺口。"""
    session = LearningSession("机会成本", "段一\n\n段二")
    cid = session.begin()
    session.record_reading_signal("confused", 1)
    assert session.learner_state.has_gap() is True
    assert "阅读原文第 2 段时用户标了「没看懂」" in session.learner_state.uncertain
    row = database.get_concept(cid)["signals"]
    import json as _json

    payload = _json.loads(row)
    assert any(s["kind"] == "confused" for s in payload["reading_signals"])


def test_clear_and_match_signals_do_not_create_gaps() -> None:
    """V0.3.0 patch 1 — 只有「没看懂」才进 Learner State，💡/✓ 不制造缺口。"""
    session = LearningSession("机会成本", "段一")
    session.begin()
    session.record_reading_signal("match", 0)
    session.record_reading_signal("clear", 0)
    assert session.stuck_points == []
    assert session.learner_state.uncertain == []
    assert session.learner_state.has_gap() is False


def test_reading_signals_ignore_blank_stuck_point() -> None:
    session, _ = make_session()
    cid = session.begin()
    session.record_stuck_point("   ")
    assert session.stuck_points == []
    assert not database.get_concept(cid)["signals"]


def test_confidence_prediction_recorded_and_calibrated() -> None:
    """验证前信心预测：记录 attempt/prediction，判级后回填 actual_level。"""
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", understood=["核心"], quality="deep"),
    )
    cid = session.begin()
    session.start_validation()
    session.record_confidence_prediction("😊 应该可以讲清楚")

    assert session.confidence_predictions == [
        {"attempt": 1, "prediction": "😊 应该可以讲清楚"}
    ]
    assert session.should_ask_confidence() is False  # 记录后不再重复追问

    session.submit_validation("回答")
    assert session.confidence_predictions[-1]["actual_level"] == "relationship"

    payload = json.loads(database.get_concept(cid)["signals"])
    assert payload["confidence"][-1]["attempt"] == 1
    assert payload["confidence"][-1]["actual_level"] == "relationship"


def test_should_ask_confidence_is_boolean() -> None:
    s1, _ = make_session()
    s1.begin()
    assert isinstance(s1.should_ask_confidence(), bool)


def test_intervention_feedback_recorded_in_history() -> None:
    """干预后反馈：写入 feedback 列表、learner_state.intervention_history，且可恢复。"""
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", uncertain=["边界不清"]),
        intervention_reply("counterexample", "如果放弃三个选择，机会成本是三个加起来吗？"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("回答")
    assert session.stage == "intervention"

    session.record_intervention_feedback("unclear")
    assert session.intervention_feedback_list == [
        {"action": "counterexample", "feedback": "unclear"}
    ]
    assert (
        session.learner_state.intervention_history[-1]["feedback"] == "unclear"
    )
    assert session.feedback_pending() is False  # 已反馈过，不再追问

    row = database.get_concept(cid)
    assert json.loads(row["intervention_feedback"]) == [
        {"action": "counterexample", "feedback": "unclear"}
    ]

    restored = restore_session(cid)
    assert restored.intervention_feedback_list == [
        {"action": "counterexample", "feedback": "unclear"}
    ]
    # 恢复后 `_current_intervention` 重置为 None，因此不再询问该反馈
    assert restored.feedback_pending() is False


def test_decider_prompt_uses_past_intervention_feedback() -> None:
    """干预决策器收到过往反馈，优先避开被说「还是有点懵」的方式。"""
    session, transport = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", uncertain=["边界不清"]),
        intervention_reply("counterexample", "反例：三个选择怎么算？"),
        update_reply("relationship", uncertain=["边界不清"], next_best_action="question"),
        intervention_reply("example", "例子：你想买电脑时的取舍"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("回答")
    session.record_intervention_feedback("unclear")
    session.submit_intervention_answer("我的回答")

    contents = [p["messages"][1]["content"] for p in transport.requests]
    deciders = [
        c for c in contents if "最小干预决策器" in c and "用户对已给干预的反馈" in c
    ]
    assert deciders, "期待决策器收到干预反馈历史"
    assert "counterexample" in deciders[-1]
    assert "unclear" in deciders[-1]


# ------------------------------------------------- V0.3.0 patch 2/3 干预闭环


def test_ineffective_intervention_records_and_escalates_floor() -> None:
    """干预无效：记录 effective=False、连续计数 +1，且下一步被强升到下一优先级。

    AI 违抗（仍返回 hint）也会被硬性抬到 example（>= 最低允许级别）。
    """
    session, transport = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", uncertain=["边界不清"]),
        intervention_reply("hint", "提示：成本指的是什么"),
        update_reply("relationship", uncertain=["边界不清"], next_best_action="question"),
        intervention_reply("hint", "提示：再想想边界"),  # 低于最低级别，应被抬升
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("回答")
    assert session.current_intervention()["action"] == "hint"

    result = session.submit_intervention_answer("我的回答")
    assert result["stage"] == "intervention"
    assert session.stage == "intervention"
    # 记录无效 + 计数
    assert session._consecutive_ineffective == 1
    assert session.learner_state.intervention_history[-2]["effective"] is False
    assert session.learner_state.intervention_history[-2]["action"] == "hint"
    # 下一次决策被强制升级（AI 返回 hint → 抬到 example）
    assert session.current_intervention()["action"] == "example"

    # 决策器 prompt 带上了「最低允许级别」约束
    decider_prompts = [
        p["messages"][1]["content"]
        for p in transport.requests
        if "最小干预决策器" in p["messages"][1]["content"]
    ]
    assert "上次干预未让用户进步" in decider_prompts[-1]
    assert "example（具体例子）" in decider_prompts[-1]
    # 该约束不落库丢失：intervention_history 快照里带 effective 标记
    restored = restore_session(cid, client=session.client)
    assert restored.learner_state.intervention_history[-1]["effective"] is False
    assert restored.learner_state.intervention_history[-1]["action"] == "hint"


def test_two_ineffective_interventions_finish() -> None:
    """连续 2 次干预无效 → 直接 complete，不再继续。"""
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", uncertain=["边界不清"]),
        intervention_reply("hint", "提示1"),
        update_reply("relationship", uncertain=["边界不清"], next_best_action="question"),
        intervention_reply("question", "提问：边界到底怎么算？"),  # floor=example，question 以更强级别继续
        update_reply("relationship", uncertain=["边界不清"], next_best_action="question"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("回答")
    session.submit_intervention_answer("答1")
    assert session.stage == "intervention"
    assert session._consecutive_ineffective == 1

    result = session.submit_intervention_answer("答2")
    assert result["stage"] == "complete"
    assert session.stage == "complete"
    assert session.phase == "connections"
    assert "final_note" in result
    assert session._consecutive_ineffective == 2
    # 进入复习队列
    row = database.get_concept(cid)
    assert row["next_review_date"] == (date.today() + timedelta(days=1)).isoformat()


def test_effective_intervention_resets_counter() -> None:
    """层级有提升 → 记录 effective=True 并清零连续无效计数。"""
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", uncertain=["边界不清"]),
        intervention_reply("hint", "提示"),
        update_reply("application", understood=["会用"], next_best_action="none"),
    )
    session.begin()
    session.start_validation()
    session.submit_validation("回答")

    result = session.submit_intervention_answer("我举了一个生活例子")
    assert result["stage"] == "complete"
    assert session._consecutive_ineffective == 0
    assert (
        session.learner_state.intervention_history[-1]["effective"] is True
    )
    assert (
        session.learner_state.understanding_level == "application"
    )


def test_escalation_caps_at_top_of_ladder() -> None:
    """阶梯顶部（question）无效时无更高级别可升级 → 尊重 AI 的收尾决策。"""
    session, _ = make_session(
        validation_task_reply("任务"),
        analysis_reply("relationship", uncertain=["边界不清"]),
        intervention_reply("question", "提问1"),
        update_reply("relationship", uncertain=["边界不清"], next_best_action="question"),
        intervention_reply("none", "到这里吧，明天复习再推进。", requires_user_response=False),
    )
    session.begin()
    session.start_validation()
    session.submit_validation("回答")
    assert session.current_intervention()["action"] == "question"

    result = session.submit_intervention_answer("答1")
    # question 之上没有更高级别 → minimum_action 为空 → AI 的 none 收尾生效
    assert result["stage"] == "complete"
    assert "到这里吧" in result["final_note"]
