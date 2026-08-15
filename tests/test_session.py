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
from core.session import LearningSession, SessionError, warmup_concept

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


def validation_task_reply(task: str, target: str) -> httpx.Response:
    return text_reply(
        json.dumps({"task": task, "target": target}, ensure_ascii=False)
    )


def validate_reply(is_correct: bool, feedback: str, missing=None) -> httpx.Response:
    body = {"is_correct": is_correct, "feedback": feedback, "missing": missing}
    return text_reply(json.dumps(body, ensure_ascii=False))


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


def test_start_validation_designs_task_and_sets_stage() -> None:
    session, transport = make_session(
        text_reply("Q1"),
        validation_task_reply("用一句话向朋友解释机会成本", "关键点：放弃的次优价值"),
    )
    session.start()
    task_text = session.start_validation()
    assert task_text == "用一句话向朋友解释机会成本"
    assert session.validation_target == "关键点：放弃的次优价值"
    assert session.stage == "validation"
    assert session.validation_passed is False
    assert session.needs_relearning is False
    payload = transport.requests[1]["messages"][1]["content"]
    assert "验证任务" in payload


def test_validation_success_sets_passed() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        validation_task_reply("任务", "目标"),
        validate_reply(True, "很好，你已经抓住了核心"),
    )
    session.start()
    session.start_validation()
    result = session.submit_validation("机会成本就是我必须放弃的那个次优选择")
    assert result["passed"] is True
    assert result["feedback"] == "很好，你已经抓住了核心"
    assert result["missing"] is None
    assert result["needs_relearning"] is False
    assert session.validation_passed is True
    assert session.validation_attempts == 0
    # 验证通过后进入深化追问阶段
    assert session.stage == "deepening"


def test_validation_three_failures_marks_needs_relearning() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        validation_task_reply("任务", "目标"),
        validate_reply(False, "再想想", "缺关键点1"),
        validate_reply(False, "再想想", "缺关键点2"),
        validate_reply(False, "最后提示", "还是缺关键点3"),
    )
    session.start()
    session.start_validation()
    r1 = session.submit_validation("回答1")
    assert r1["passed"] is False
    assert r1["attempts_left"] == 2
    assert r1["needs_relearning"] is False

    r2 = session.submit_validation("回答2")
    assert r2["attempts_left"] == 1
    assert r2["needs_relearning"] is False

    r3 = session.submit_validation("回答3")
    assert r3["needs_relearning"] is True
    assert r3["stage"] == "relearn"
    assert session.stage == "relearn"
    assert session.needs_relearning is True


def test_submit_validation_requires_validation_stage() -> None:
    session, _ = make_session(text_reply("Q1"))
    session.start()
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
