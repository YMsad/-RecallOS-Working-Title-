"""Tests for the learning session flow (layers, hints, references, summary)."""

from __future__ import annotations

import json

import httpx
import pytest

from core import database
from core.client import DeepSeekClient
from core.config import Settings
from core.models import MASTERY_UNCLEAR, MASTERY_UNDERSTOOD
from core.session import LearningSession, SessionError

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
                 source="原文：选择意味着放弃", max_fail=3):
    transport = ScriptedTransport(list(responses))
    client = DeepSeekClient(
        settings=TEST_SETTINGS, transport=httpx.MockTransport(transport.handler)
    )
    session = LearningSession(
        title, source, client=client, max_consecutive_failures=max_fail
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
    )
    session.start()
    result = session.submit_answer("错误回答")
    assert result["correct"] is False
    assert result["hint"] == "想一想你放弃了什么"
    assert result["reference"] is None
    assert result["is_done"] is False
    assert session.layer == 1
    assert session.next_question() == "Q1"
    history = database.get_qa_history(session.concept_id)
    assert len(history) == 1
    assert history[0]["hint_used"] == 1
    assert history[0]["is_correct"] == 0


def test_three_failures_give_reference_and_mark_uncertain() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        judge(False, "再想想", "h1"),
        judge(False, "再想想", "h2"),
        judge(False, "再想想", "h3"),
        text_reply("参考解释：机会成本是你放弃的次优价值"),
        text_reply("Q2"),
    )
    session.start()
    r1 = session.submit_answer("A1")
    r2 = session.submit_answer("A2")
    assert r1["reference"] is None and r2["reference"] is None
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
        judge(False, "再想想", "h"),
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
