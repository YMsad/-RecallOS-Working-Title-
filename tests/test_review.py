"""Tests for the review queue and ReviewSession (V0.2.2)."""

from __future__ import annotations

import json
from datetime import date, timedelta

import httpx
import pytest

from core import database
from core.client import DeepSeekClient
from core.config import Settings
from core.models import MASTERY_LEARNING, MASTERY_UNCLEAR, MASTERY_UNDERSTOOD
from core.review import (
    MAX_REVIEW_ATTEMPTS,
    ReviewSession,
    add_to_review_queue,
    get_due_reviews,
    update_review_status,
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
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        assert self.responses, "no more scripted responses"
        return self.responses.pop(0)


def text_reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def judge(is_correct: bool, feedback: str) -> httpx.Response:
    body = {"is_correct": is_correct, "feedback": feedback, "hint": None}
    return text_reply(json.dumps(body, ensure_ascii=False))


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    database.configure(tmp_path / "test.db")
    yield


def make_client(*responses: httpx.Response) -> tuple[DeepSeekClient, ScriptedTransport]:
    transport = ScriptedTransport(list(responses))
    client = DeepSeekClient(
        settings=TEST_SETTINGS, transport=httpx.MockTransport(transport.handler)
    )
    return client, transport


def seed_concept(title="机会成本", *, mastery=MASTERY_UNDERSTOOD) -> int:
    cid = database.save_concept(title, "原文：选择意味着放弃")
    database.update_concept(cid, mastery=mastery)
    return cid


# ------------------------------------------------------------------ queue


def test_add_to_review_queue_sets_tomorrow() -> None:
    cid = seed_concept()
    assert database.get_concept(cid)["next_review_date"] is None
    assert add_to_review_queue(cid) is True
    expected = (date.today() + timedelta(days=1)).isoformat()
    assert database.get_concept(cid)["next_review_date"] == expected


def test_get_due_reviews_returns_overdue_and_today() -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    c1 = seed_concept("概念A")
    c2 = seed_concept("概念B")
    c3 = seed_concept("概念C")
    database.update_concept(c1, next_review_date=tomorrow)
    database.update_concept(c2, next_review_date=yesterday)
    database.update_concept(c3, next_review_date=tomorrow)

    due = get_due_reviews()
    titles = [c["title"] for c in due]
    assert titles == ["概念B"]  # only overdue today


def test_update_review_status_passed_advances_stage() -> None:
    cid = seed_concept()
    assert update_review_status(cid, passed=True) is True
    row = database.get_concept(cid)
    assert row["review_stage"] == 1
    assert row["review_count"] == 1
    assert row["mastery"] == MASTERY_UNDERSTOOD
    gap = min(7, 1 << 1)
    assert row["next_review_date"] == (date.today() + timedelta(days=gap)).isoformat()

    assert update_review_status(cid, passed=True) is True
    row = database.get_concept(cid)
    assert row["review_stage"] == 2
    assert row["review_count"] == 2


def test_update_review_status_failed_resets_to_tomorrow() -> None:
    cid = seed_concept()
    update_review_status(cid, passed=True)
    assert update_review_status(cid, passed=False) is True
    row = database.get_concept(cid)
    assert row["review_stage"] == 0
    assert row["review_count"] == 2
    assert row["mastery"] == MASTERY_UNCLEAR
    assert row["next_review_date"] == (date.today() + timedelta(days=1)).isoformat()


def test_update_review_status_missing_concept_returns_false() -> None:
    assert update_review_status(99999, passed=True) is False


# ---------------------------------------------------------------- session


def test_review_session_uses_tomorrow_hook_from_summary() -> None:
    cid = seed_concept()
    database.save_daily_summary(cid, "搞懂了", "边际效用和机会成本是什么关系？")
    client, _ = make_client(judge(True, "记得很清楚"))
    session = ReviewSession(cid, client=client)
    question = session.start()
    assert question == "边际效用和机会成本是什么关系？"
    assert session.next_question() == question
    assert session.phase == "reviewing"


def test_review_session_generates_question_when_no_hook() -> None:
    cid = seed_concept()
    client, transport = make_client(text_reply("复习题：什么是机会成本？"))
    session = ReviewSession(cid, client=client)
    question = session.start()
    assert question == "复习题：什么是机会成本？"
    payload = transport.requests[0]["messages"][1]["content"]
    assert "复习" in payload


def test_review_pass_advances_stage_and_logs() -> None:
    cid = seed_concept()
    database.save_daily_summary(cid, "搞懂了", "明天的追问")
    client, _ = make_client(judge(True, "回答得很好"))
    session = ReviewSession(cid, client=client)
    session.start()
    result = session.submit_answer("我的回答")
    assert result["passed"] is True
    assert result["attempts"] == 1
    assert session.phase == "finished"
    assert session.needs_relearn is False
    assert database.get_concept(cid)["review_stage"] == 1
    logs = database.get_review_logs(cid)
    assert len(logs) == 1
    assert logs[0]["passed"] == 1
    assert logs[0]["question"] == "明天的追问"


def test_review_three_failures_marks_needs_relearn() -> None:
    cid = seed_concept()
    database.save_daily_summary(cid, "搞懂了", "明天的追问")
    client, _ = make_client(*(judge(False, "再想想") for _ in range(MAX_REVIEW_ATTEMPTS)))
    session = ReviewSession(cid, client=client)
    session.start()
    for i in range(MAX_REVIEW_ATTEMPTS):
        result = session.submit_answer(f"答{i + 1}")
        assert result["passed"] is False
        assert result["attempts"] == i + 1
    assert session.phase == "finished"
    assert session.needs_relearn is True
    assert session.last_result["needs_relearn"] is True
    row = database.get_concept(cid)
    assert row["mastery"] == MASTERY_LEARNING
    assert row["review_stage"] == 0
    logs = database.get_review_logs(cid)
    assert len(logs) == MAX_REVIEW_ATTEMPTS


def test_review_submit_before_start_raises() -> None:
    cid = seed_concept()
    client, _ = make_client(text_reply("复习题"), judge(True, "对"))
    session = ReviewSession(cid, client=client)
    with pytest.raises(ValueError):
        session.submit_answer("x")
    session.start()
    session.submit_answer("答")
    assert session.phase == "finished"
