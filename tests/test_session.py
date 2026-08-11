"""Tests for the learning session flow (layers, hints, references, summary)."""

from __future__ import annotations

import json

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


def test_route_beginner_uses_scenario() -> None:
    session, _ = make_session(text_reply("Q1"))
    assert session._route_model(1) == "scenario"
    assert session._route_model(2) == "scenario"
    assert session._route_model(3) == "scenario"


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
