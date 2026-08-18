"""Tests for the learning session flow (layers, hints, references, summary)."""

from __future__ import annotations

import json
from datetime import date, timedelta

import httpx
import pytest

from core import database
from core.client import DeepSeekClient, DeepSeekError
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
    action="hint", content='Think about what "cost" refers to', requires_user_response=True, reason=None
) -> httpx.Response:
    body = {
        "action": action,
        "content": content,
        "requires_user_response": requires_user_response,
        "reason": reason,
    }
    return text_reply(json.dumps(body, ensure_ascii=False))


def offer_reply(offer: str = "Want to go one layer deeper?") -> httpx.Response:
    return text_reply(
        json.dumps({"offer": offer, "options": ["Go deeper", "That's enough"]}, ensure_ascii=False)
    )


def feedback_reply(text: str = "Your answer mentioned: ...") -> httpx.Response:
    """Validation feedback (V0.3.1 hotfix) — plain text, not validated as JSON."""
    return text_reply(text)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    database.configure(tmp_path / "test.db")
    yield


def make_session(*responses: httpx.Response, title="opportunity cost",
                 source="Original: choosing means giving up", max_fail=3, mode="beginner"):
    transport = ScriptedTransport(list(responses))
    client = DeepSeekClient(
        settings=TEST_SETTINGS, transport=httpx.MockTransport(transport.handler)
    )
    session = LearningSession(
        title, source, client=client, max_consecutive_failures=max_fail, mode=mode
    )
    return session, transport


def test_start_saves_concept_and_returns_question() -> None:
    session, _ = make_session(text_reply("Q1: What is the core of opportunity cost?"))
    question = session.start()
    assert question == "Q1: What is the core of opportunity cost?"
    assert session.layer == 1
    assert session.phase == "learning"
    row = database.get_concept(session.concept_id)
    assert row["title"] == "opportunity cost"
    assert row["mastery"] == "Learning"


def test_begin_saves_concept_and_enters_reading_stage() -> None:
    """V0.3.0 — new begin() flow: only saves the concept, no opening question, enters the reading stage."""
    session, transport = make_session()
    cid = session.begin()
    assert cid is not None
    assert transport.requests == []  # no AI calls, no scripts consumed
    assert session.stage == "reading"
    assert session.flow == "legacy"
    assert session.phase == "learning"
    assert database.get_concept(cid)["title"] == "opportunity cost"
    # idempotent: calling again returns the same concept_id
    assert session.begin() == cid


def test_full_flow_all_correct() -> None:
    database.save_concept("sunk cost", "A concept learned before")
    session, transport = make_session(
        text_reply("Q1"),
        judge(True, "Right"),
        text_reply("Q2"),
        judge(True, "Right"),
        text_reply("Q3"),
        judge(True, "Right"),
        text_reply("Q4"),
        judge(True, "Right"),
        connections_reply(
            {"concept_title": "sunk cost", "relation_text": "both about choice, one looks to the future and the other to the past"}
        ),
        summary_reply("I finally understood opportunity cost", "What is the relationship between marginal utility and opportunity cost?"),
    )
    session.start()
    for _ in range(4):
        result = session.submit_answer("my answer")
        assert result["correct"] is True

    assert session.phase == "connections"
    assert session.next_question() is None

    # Layer 4 question was seeded with previously learned concepts
    layer4_payload = next(
        p for p in transport.requests if "Layer 4" in p["messages"][1]["content"]
    )
    assert "sunk cost" in layer4_payload["messages"][1]["content"]

    conns = session.get_connections()
    assert len(conns) == 1
    assert conns[0].concept_title == "sunk cost"
    assert len(database.get_connections(session.concept_id)) == 1

    summary = session.finish(user_definition="Opportunity cost is the value you give up")
    assert summary.breakthrough == "I finally understood opportunity cost"
    assert session.phase == "finished"

    today = database.get_today_summary()
    assert today["breakthrough_text"] == "I finally understood opportunity cost"
    assert today["tomorrow_hook"] == "What is the relationship between marginal utility and opportunity cost?"
    row = database.get_concept(session.concept_id)
    assert row["mastery"] == MASTERY_UNDERSTOOD
    assert row["user_definition"] == "Opportunity cost is the value you give up"


def test_wrong_answer_returns_hint_and_stays() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        judge(False, "Think again", "Think about what you gave up"),
        text_reply("Simpler question: what did you give up when you bought bubble tea?"),
    )
    session.start()
    result = session.submit_answer("wrong answer")
    assert result["correct"] is False
    assert result["hint"] == "Think about what you gave up"
    assert result["reference"] is None
    assert result["is_done"] is False
    assert result["simplified"] is True
    assert session.layer == 1
    # dimension reduction: a simpler question instead of the original one
    assert session.next_question() == "Simpler question: what did you give up when you bought bubble tea?"
    history = database.get_qa_history(session.concept_id)
    assert len(history) == 1
    assert history[0]["hint_used"] == 1
    assert history[0]["is_correct"] == 0


def test_three_failures_escalate_simplify_angle_reference() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        judge(False, "Think again", "h1"),
        text_reply("simplified question"),
        judge(False, "Think again", "h2"),
        text_reply("question from a different angle"),
        judge(False, "Think again", "h3"),
        text_reply("Reference explanation: opportunity cost is the second-best value you gave up"),
        text_reply("Q2"),
    )
    session.start()
    r1 = session.submit_answer("A1")
    assert r1["simplified"] is True
    assert r1["angle_shift"] is False
    assert r1["reference"] is None
    assert session.next_question() == "simplified question"

    r2 = session.submit_answer("A2")
    assert r2["angle_shift"] is True
    assert r2["reference"] is None
    assert session.next_question() == "question from a different angle"

    r3 = session.submit_answer("A3")
    assert r3["reference"] == "Reference explanation: opportunity cost is the second-best value you gave up"
    assert r3["mastery"] == MASTERY_UNCLEAR
    assert r3["is_done"] is False
    assert session.layer == 2
    assert session.next_question() == "Q2"
    assert database.get_concept(session.concept_id)["mastery"] == MASTERY_UNCLEAR


def test_uncertain_flow_finishes_with_mastery_uncertain() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        judge(False, "Think again", "h"),
        text_reply("simplified"),
        judge(False, "Think again", "h"),
        text_reply("different angle"),
        judge(False, "Think again", "h"),
        text_reply("reference explanation"),
        text_reply("Q2"),
        judge(True, "Right"),
        text_reply("Q3"),
        judge(True, "Right"),
        text_reply("Q4"),
        judge(True, "Right"),
        connections_reply({"concept_title": "sunk cost", "relation_text": "R"}),
        summary_reply("Learned something", "Keep thinking tomorrow"),
    )
    session.start()
    session.submit_answer("wrong")
    session.submit_answer("wrong")
    session.submit_answer("wrong")
    session.submit_answer("right")
    session.submit_answer("right")
    session.submit_answer("right")
    assert session.phase == "connections"
    session.get_connections()
    session.finish(user_definition="I kind of get it now")
    assert database.get_concept(session.concept_id)["mastery"] == MASTERY_UNCLEAR


def test_explain_mode_returns_plain_text_and_resets_failures() -> None:
    session, transport = make_session(
        text_reply("Q1"),
        judge(False, "Think again", "h"),
        text_reply("simplified"),
        judge(False, "Think again", "h"),
        text_reply("question from a different angle"),
        text_reply("Plain-language explanation: opportunity cost is the second-best choice you gave up"),
    )
    session.start()
    session.submit_answer("wrong")
    session.submit_answer("wrong again")
    assert session.consecutive_failures == 2
    explanation = session.explain()
    assert explanation == "Plain-language explanation: opportunity cost is the second-best choice you gave up"
    assert session.explain_used is True
    assert session.consecutive_failures == 0
    explain_payload = next(
        p for p in transport.requests if "I don't get it" in p["messages"][1]["content"]
    )
    assert "plain language" in explain_payload["messages"][1]["content"]


def test_ask_for_angle_switch_explicit() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        text_reply("question from a different angle"),
    )
    session.start()
    question = session.ask_for_angle_switch()
    assert question == "question from a different angle"
    assert session.consecutive_failures == 0
    assert session.next_question() == "question from a different angle"


def test_opening_question_is_tailored_to_level_and_interest() -> None:
    session, transport = make_session(text_reply("opening question"))
    session.start()
    opening_payload = transport.requests[0]["messages"][1]["content"]
    assert "very first question" in opening_payload
    assert "Learner's baseline" in opening_payload
    assert "interest" in opening_payload


def test_warmup_returns_plain_text_for_beginner_mode() -> None:
    session, transport = make_session(text_reply("Opportunity cost is the B you give up to get A."))
    warmup = session.warmup()
    assert warmup == "Opportunity cost is the B you give up to get A."
    warmup_payload = transport.requests[0]["messages"][1]["content"]
    assert "warm-up" in warmup_payload
    assert "In one sentence" in warmup_payload


def test_warmup_skipped_for_advanced_mode() -> None:
    session, _ = make_session()
    session.mode = "advanced"
    assert session.warmup() == ""
    assert session._current_question is None


def test_warmup_concept_standalone_function() -> None:
    transport = ScriptedTransport(
        [text_reply("Opportunity cost is the B you give up to get A.")]
    )
    client = DeepSeekClient(
        settings=TEST_SETTINGS, transport=httpx.MockTransport(transport.handler)
    )
    warmup = warmup_concept("opportunity cost", "Original: choosing means giving up", client=client)
    assert warmup == "Opportunity cost is the B you give up to get A."
    payload = transport.requests[0]["messages"][1]["content"]
    assert "warm-up" in payload
    assert "In one sentence" in payload


def test_cognitive_contrast_used_in_beginner_layers() -> None:
    session, transport = make_session(
        text_reply("Q1"),
        judge(True, "Right"),
        text_reply("Q2"),
        judge(True, "Right"),
        text_reply("Q3"),
        judge(True, "Right"),
        text_reply("Q4"),
        judge(True, "Right"),
    )
    session.start()
    for _ in range(3):
        session.submit_answer("answer")
    layer2_payload = next(
        p for p in transport.requests if "Layer 2" in p["messages"][1]["content"]
    )
    assert "Cognitive contrast" in layer2_payload["messages"][1]["content"]
    layer3_payload = next(
        p for p in transport.requests if "Layer 3" in p["messages"][1]["content"]
    )
    assert "Cognitive contrast" in layer3_payload["messages"][1]["content"]
    layer4_payload = next(
        p for p in transport.requests if "Layer 4" in p["messages"][1]["content"]
    )
    assert "Cognitive contrast" not in layer4_payload["messages"][1]["content"]


def test_cognitive_contrast_not_used_in_advanced_mode() -> None:
    session, transport = make_session(
        text_reply("Q1"),
        judge(True, "Right"),
        text_reply("Q2"),
    )
    session.mode = "advanced"
    session.start()
    session.submit_answer("answer")
    layer2_payload = next(
        p for p in transport.requests if "Layer 2" in p["messages"][1]["content"]
    )
    assert "Cognitive contrast" not in layer2_payload["messages"][1]["content"]


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
        judge(True, "Right"),
        text_reply("Q2"),
        judge(True, "Right"),
        text_reply("Q3"),
        judge(True, "Right"),
        text_reply("Q4"),
        judge(True, "Right"),
        connections_reply({"concept_title": "Nonexistent concept", "relation_text": "R"}),
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
        text_reply("Q1"), judge(True, "Right"), text_reply("Q2"), mode="advanced"
    )
    session.start()
    session.submit_answer("answer")
    payload = next(
        p for p in transport.requests if "Layer 2" in p["messages"][1]["content"]
    )
    assert "Thinking model [First principles]" in payload["messages"][1]["content"]


# ------------------------------------------------------- V0.2.3 review queue


def test_learning_complete_queues_review_without_finish() -> None:
    """When learning finishes (even without a summary), the concept must join the review queue."""
    session, _ = make_session(
        text_reply("Q1"),
        judge(True, "Right"),
        text_reply("Q2"),
        judge(True, "Right"),
        text_reply("Q3"),
        judge(True, "Right"),
        text_reply("Q4"),
        judge(True, "Right"),
    )
    session.start()
    for _ in range(4):
        session.submit_answer("answer")
    assert session.phase == "connections"
    row = database.get_concept(session.concept_id)
    expected = (date.today() + timedelta(days=1)).isoformat()
    assert row["next_review_date"] == expected


# ------------------------------------------------------- V0.3.0 validation stage
# Learning Loop v2: validation task → learner state analysis → minimal intervention → dynamic finish


def test_start_validation_designs_task_and_sets_stage() -> None:
    session, transport = make_session(
        validation_task_reply("Explain opportunity cost to a friend in one sentence", "summary", 2),
    )
    cid = session.begin()
    task_text = session.start_validation()
    assert task_text == "Explain opportunity cost to a friend in one sentence"
    assert session.validation_kind == "summary"
    assert session.validation_difficulty == 2
    assert session.stage == "validation"
    assert session.validation_passed is False
    assert session.needs_relearning is False
    payload = transport.requests[0]["messages"][1]["content"]
    assert "validation" in payload
    assert '"task"' in payload
    # learner state was reset
    assert session.learner_state.understanding_level == "surface"
    assert session.current_intervention() is None
    assert session._offer is None


def test_start_validation_malformed_reply_raises_friendly_session_error() -> None:
    """AI returns non-JSON: converted to a retryable SessionError, not an uncaught validation exception."""
    session, _ = make_session(text_reply("Sorry, I didn't output in the expected format this time"))
    session.begin()
    with pytest.raises(SessionError, match="not in the expected format"):
        session.start_validation()
    assert session.stage == "reading"  # failure doesn't advance


def test_submit_validation_no_gap_completes() -> None:
    """V0.3.1 — validation passed finishes: no gap → complete immediately, no offer."""
    session, _ = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", understood=["understands the core meaning"], quality="deep"),
    )
    cid = session.begin()
    session.start_validation()
    result = session.submit_validation("Opportunity cost is the second-best alternative I must give up")
    assert result["stage"] == "complete"
    assert "Your understanding is solid" in result["final_note"]
    assert session.validation_passed is True
    assert session.validation_attempts == 0
    assert session.stage == "complete"
    assert session.phase == "connections"
    assert len(session.validation_history) == 1
    assert session.learner_state.has_gap() is False
    # completing joins the review queue
    row = database.get_concept(cid)
    assert row["next_review_date"] == (date.today() + timedelta(days=1)).isoformat()


def test_submit_validation_gap_moves_to_intervention() -> None:
    session, transport = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", uncertain=["boundary unclear"], misconceptions=["confuses opportunity cost with sunk cost"]),
        intervention_reply("counterexample", "If you give up three options, is the opportunity cost all three added together?"),
        feedback_reply("**Your answer mentioned:**\n- decision\n\nBut the core of opportunity cost is the best alternative you give up."),
    )
    cid = session.begin()
    session.start_validation()
    result = session.submit_validation("answer")
    assert result["stage"] == "intervention"
    assert "If you give up three options" in result["bubble"]
    # V0.3.1 hotfix — feedback first, intervention after
    assert "Your answer mentioned" in result["bubble"]
    assert result["bubble"].index("Your answer mentioned") < result["bubble"].index("If you give up three options")
    assert session.stage == "intervention"
    assert session.validation_passed is False
    assert session.current_intervention()["action"] == "counterexample"
    assert len(session.validation_history) == 1
    # decider/analyzer prompts both carry the learner state
    payloads = [p["messages"][1]["content"] for p in transport.requests]
    assert any("learner-state analyzer" in p for p in payloads)
    assert any("minimal-intervention decider" in p for p in payloads)
    assert any("Socratic mentor" in p for p in payloads)


def test_validation_feedback_failure_falls_back_without_blocking() -> None:
    """V0.3.1 hotfix — when the feedback AI fails, degrade to a template without blocking the intervention flow."""

    class FailingFeedbackClient:
        def __init__(self, inner):
            self._inner = inner

        def chat(self, messages, **kwargs):
            if "Socratic mentor" in messages[1]["content"]:
                raise DeepSeekError("simulated feedback failure")
            return self._inner.chat(messages, **kwargs)

    transport = ScriptedTransport(
        [
            validation_task_reply("task"),
            analysis_reply("relationship", uncertain=["boundary unclear"]),
            intervention_reply("counterexample", "If you give up three options, is the opportunity cost all three added together?"),
        ]
    )
    inner = DeepSeekClient(
        settings=TEST_SETTINGS, transport=httpx.MockTransport(transport.handler)
    )
    session = LearningSession(
        "opportunity cost", "Original: choosing means giving up", client=FailingFeedbackClient(inner)
    )
    cid = session.begin()
    session.start_validation()
    result = session.submit_validation("answer")
    assert result["stage"] == "intervention"
    # degraded template feedback (with the Learner State gap) + a complete intervention
    assert "Your answer covered" in result["bubble"]
    assert "Not quite there yet" in result["bubble"]
    assert "If you give up three options" in result["bubble"]
    assert session.current_intervention()["action"] == "counterexample"


def test_submit_validation_closing_note_finishes() -> None:
    """Decider returns action=none (closing note, no user response needed) → finish the whole flow."""
    session, _ = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", uncertain=["boundary unclear"]),
        intervention_reply("none", "You already understand the core; the boundary issue is no longer worth pursuing.", requires_user_response=False),
    )
    cid = session.begin()
    session.start_validation()
    result = session.submit_validation("answer")
    assert result["stage"] == "complete"
    assert "You already understand the core" in result["final_note"]
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
        text_reply("Simpler plain talk: opportunity cost is the B you give up for A"),
    )
    session.start()
    session.stage = "validation"
    simple = session.ask_simplify()
    assert simple == "Simpler plain talk: opportunity cost is the B you give up for A"
    assert session.stage == "validation"
    payload = transport.requests[1]["messages"][1]["content"]
    assert "not simple enough" in payload


# ------------------------------------------------------ V0.3.0 deepen offer loop


def test_offer_deepening_generates_offer() -> None:
    """V0.3.1 — the offer API is kept (legacy data only): must be in the offer stage first."""
    session, transport = make_session(
        validation_task_reply("task"),
        offer_reply("You've got the core — want to go one layer deeper?"),
    )
    session.begin()
    session.start_validation()
    session.stage = "offer"  # the main flow no longer enters offer; set it explicitly to test the legacy API
    result = session.offer_deepening()
    assert result["offer"] == "You've got the core — want to go one layer deeper?"
    assert result["options"] == ["Go deeper", "That's enough"]
    assert session.stage == "offer"
    payload = transport.requests[1]["messages"][1]["content"]
    assert "go deeper" in payload


def test_offer_deepening_requires_offer_stage() -> None:
    session, _ = make_session()
    session.begin()
    with pytest.raises(SessionError):
        session.offer_deepening()


def test_choose_deepening_stop_finishes() -> None:
    """V0.3.1 — under the legacy offer stage, "That's enough" still finishes the whole flow."""
    session, _ = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", understood=["core"], quality="deep"),
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
    """V0.3.1 — under the legacy offer stage, choosing to go deeper still enters the intervention loop."""
    session, _ = make_session(
        validation_task_reply("task"),
        offer_reply("Want to go one layer deeper?"),
        intervention_reply("example", "Think about the choice of giving up gaming time this weekend"),
        update_reply("application", understood=["core", "can apply it"], next_best_action="none"),
    )
    cid = session.begin()
    session.start_validation()
    session.stage = "offer"
    session.offer_deepening()
    result = session.choose_deepening(True)
    assert result["stage"] == "intervention"
    assert "Think about the choice" in result["bubble"]
    assert session.stage == "intervention"
    assert session.current_intervention()["action"] == "example"

    # user answers → updater sees a big gain (no gap + next_best_action=none) → re-decide → no intervention left → complete
    result = session.submit_intervention_answer("On weekends I weigh the trade-offs")
    assert result["stage"] == "complete"
    assert session.stage == "complete"
    assert session.phase == "connections"
    assert len(session.deeper_history) == 1
    assert session.deeper_history[0]["question"].startswith("Think about the choice")
    assert session.deeper_history[0]["answer"] == "On weekends I weigh the trade-offs"
    assert session.deeper_history[0]["understanding_level"] == "application"
    assert database.get_concept(cid)["mastery"] is not None


def test_submit_intervention_answer_requires_active_intervention() -> None:
    session, _ = make_session()
    session.begin()
    with pytest.raises(SessionError):
        session.submit_intervention_answer("x")


def test_next_intervention_continues_after_restore() -> None:
    session, _ = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", uncertain=["boundary"], misconceptions=["confusion"]),
        intervention_reply("counterexample", "If you give up three options, is the opportunity cost all three added together?"),
        feedback_reply("Your answer mentioned: boundary problem"),
        intervention_reply("question", 'In opportunity cost, what does "cost" refer to?'),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("answer")
    assert session.stage == "intervention"

    restored = restore_session(cid, client=session.client)
    assert restored.flow == "new"
    assert restored.stage == "intervention"
    assert restored.current_intervention() is None  # unanswered interventions are not persisted

    result = restored.next_intervention()
    assert result["stage"] == "intervention"
    assert "In opportunity cost" in result["bubble"]
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
    # keep the historical high water mark when the AI downgrades the level
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
        understanding_level="relationship", uncertain=["boundary"], next_best_action="none"
    ).should_stop() is False
    # no gap but the AI still thinks an action is worthwhile → don't stop
    assert LearnerState(
        understanding_level="relationship", next_best_action="hint"
    ).should_stop() is False


# -------------------------------------------------------- V0.3.0 deeper questions


def test_next_deeper_question_runs_full_sequence() -> None:
    session, transport = make_session(
        text_reply("Q1"),
        text_reply("Re-verify question"),
        text_reply("Connection question"),
        text_reply("Counterfactual question"),
        text_reply("Action question"),
        text_reply("First principles question"),
    )
    session.start()

    questions = []
    while True:
        q = session.next_deeper_question()
        if q is None:
            break
        questions.append(q)

    assert questions == [
        "Re-verify question",
        "Connection question",
        "Counterfactual question",
        "Action question",
        "First principles question",
    ]
    assert session.current_deeper_index == 5
    assert len(session.deeper_questions) == 5
    assert session.next_deeper_question() is None
    # all deeper questions asked → new flow learning complete and joins the review queue
    assert session.stage == "complete"
    assert session.phase == "connections"

    deeper_payloads = [
        p["messages"][1]["content"]
        for p in transport.requests
        if "Deeper probe" in p["messages"][1]["content"]
    ]
    assert len(deeper_payloads) == 5
    assert "Deeper probe: Re-verify" in deeper_payloads[0]
    assert "Deeper probe: Connection" in deeper_payloads[1]
    assert "Deeper probe: Counterfactual" in deeper_payloads[2]
    assert "Deeper probe: Action" in deeper_payloads[3]
    assert "Deeper probe: First principles" in deeper_payloads[4]


def test_submit_deeper_answer_records_exchange() -> None:
    session, _ = make_session(
        text_reply("Q1"),
        text_reply("Re-verify question"),
    )
    session.start()
    q = session.next_deeper_question()
    assert q == "Re-verify question"
    result = session.submit_deeper_answer("I think opportunity cost is the second-best thing you give up")
    assert result["recorded"] is True
    assert result["question"] == "Re-verify question"
    assert len(session.deeper_history) == 1
    assert session.deeper_history[0]["answer"] == "I think opportunity cost is the second-best thing you give up"


def test_submit_deeper_answer_requires_open_question() -> None:
    session, _ = make_session(text_reply("Q1"))
    session.start()
    with pytest.raises(SessionError):
        session.submit_deeper_answer("x")


# ------------------------------------------------------- V0.3.0 session restore


def test_restore_legacy_flow_rebuilds_qa_history() -> None:
    """Concepts started in the old flow (no stage/validation_type markers) still restore as the old flow."""
    session, _ = make_session(text_reply("Q1"), judge(True, "Right"), text_reply("Q2"))
    session.start()
    session.submit_answer("answer")
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
    assert restored.title == "opportunity cost"


def test_restore_new_flow_validation_preserves_state() -> None:
    session, _ = make_session(
        validation_task_reply("task", "summary", 2),
        analysis_reply("relationship", uncertain=["boundary"]),
        intervention_reply("hint", 'Think about what "cost" refers to'),
        feedback_reply("Your answer mentioned: ..."),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("answer 1")
    assert session.stage == "intervention"

    restored = restore_session(cid)
    assert restored.flow == "new"
    assert restored.stage == "intervention"
    assert restored.validation_task == "task"
    assert restored.validation_kind == "summary"
    assert restored.validation_difficulty == 2
    assert restored.validation_attempts == 1
    assert restored.current_intervention() is None  # unanswered interventions are not persisted
    assert len(restored.validation_history) == 1
    assert restored.validation_history[0]["understanding_level"] == "relationship"
    assert restored.learner_state.understanding_level == "relationship"


def test_restore_new_flow_offer_stage() -> None:
    """V0.3.1 — legacy offer-stage data (from older versions) still restores and lets the UI wrap up."""
    session, _ = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", understood=["core"], quality="deep"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("answer")
    assert session.stage == "complete"  # the main flow no longer generates an offer
    # simulate old data: flip the stage back to offer and persist
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
        validation_task_reply("task"),
        analysis_reply("relationship", understood=["core"], quality="deep"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("answer")
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
    """V0.3.1 — finishing auto-generates the summary: uses the latest validation answer, persists and updates mastery."""
    session, transport = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", understood=["understands the core meaning"], quality="deep"),
        summary_reply("I finally understood opportunity cost", "Tomorrow, think of a counterexample"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("Opportunity cost is the best choice I gave up")
    assert session.stage == "complete"

    summary = session.finish_auto()
    assert summary.breakthrough == "I finally understood opportunity cost"
    assert summary.tomorrow_hook == "Tomorrow, think of a counterexample"
    assert session.phase == "finished"

    row = database.get_concept(cid)
    assert row["mastery"] == MASTERY_UNDERSTOOD
    assert row["user_definition"] == "Opportunity cost is the best choice I gave up"
    today = database.get_today_summary()
    assert today is not None
    assert today["breakthrough_text"] == "I finally understood opportunity cost"
    assert today["tomorrow_hook"] == "Tomorrow, think of a counterexample"
    payload = transport.requests[2]["messages"][1]["content"]
    assert "Opportunity cost is the best choice I gave up" in payload


def test_finish_auto_requires_complete_stage() -> None:
    session, _ = make_session()
    session.begin()
    with pytest.raises(SessionError, match="complete"):
        session.finish_auto()


def test_record_reading_answer_persists_and_restores() -> None:
    """V0.3.1 fix — reading answers are recorded per paragraph, rewritten on the same paragraph, persisted and restorable."""
    session, _ = make_session(source="Paragraph one\n\nParagraph two")
    cid = session.begin()
    session.record_reading_answer(0, "Paragraph one is about the definition of opportunity cost")
    session.record_reading_answer(1, "Paragraph two is about the cost of a choice")
    session.record_reading_answer(0, "Opportunity cost is the second-best choice you give up")

    saved = json.loads(database.get_concept(cid)["reading_answers"])
    assert [e["paragraph_index"] for e in saved] == [0, 1]
    assert saved[0]["answer"] == "Opportunity cost is the second-best choice you give up"
    assert session.reading_answer_count() == 2

    restored = restore_session(cid)
    assert restored.reading_answer_text(0) == "Opportunity cost is the second-best choice you give up"
    assert restored.reading_answer_text(1) == "Paragraph two is about the cost of a choice"
    assert restored.reading_answers == session.reading_answers


def test_reading_answers_feed_finish_auto_prompt() -> None:
    """V0.3.1 fix — reading answers go into the summary prompt as context, supplementing the validation answer."""
    session, transport = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", understood=["core"], quality="deep"),
        summary_reply("I finally understood opportunity cost", "Tomorrow, think of a counterexample"),
    )
    cid = session.begin()
    session.record_reading_answer(0, "Opportunity cost is the option you give up")
    session.start_validation()
    session.submit_validation("Choosing A means losing B")
    assert session.stage == "complete"

    session.finish_auto()
    assert session.phase == "finished"
    payload = transport.requests[2]["messages"][1]["content"]
    assert "Opportunity cost is the option you give up" in payload  # reading answer in prompt
    assert "Choosing A means losing B" in payload
    assert database.get_concept(cid)["user_definition"] == "Choosing A means losing B"


def test_record_reading_answer_fallback_for_definition() -> None:
    """V0.3.1 fix — without a validation answer, reading answers fall back to "my understanding"."""
    session, _ = make_session(
        summary_reply("I finally understood opportunity cost", "Tomorrow, think of a counterexample"),
    )
    cid = session.begin()
    session.record_reading_answer(0, "Opportunity cost is the second-best choice you give up")
    session.stage = "complete"
    session.finish_auto()
    assert database.get_concept(cid)["user_definition"] == "Opportunity cost is the second-best choice you give up"


# ------------------------------------------------------ V0.3.1 user signals


def test_learning_goal_stored_and_restored() -> None:
    """The learning goal is optional; an invalid value falls back to understand, and it restores with the session."""
    session = LearningSession("opportunity cost", "Original text", learning_goal="apply")
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
    """Reading-stage 🤔💡✓ markers and stuck points: recorded, persisted, restored."""
    session = LearningSession("opportunity cost", "Paragraph one\n\nParagraph two\n\nParagraph three")
    cid = session.begin()
    session.record_reading_signal("confused", 0)
    session.record_reading_signal("clear", 2)
    session.record_stuck_point("Does opportunity cost only count the best option, or all of them?")

    assert session.reading_signals == [
        {"kind": "confused", "position": 0},
        {"kind": "clear", "position": 2},
    ]
    # V0.3.0 patch 1 — marking "didn't get it" auto-fills stuck_points / learner_state.uncertain
    derived = 'The learner marked paragraph 1 as "didn\'t get it" while reading'
    assert session.stuck_points == [derived, "Does opportunity cost only count the best option, or all of them?"]
    assert session.learner_state.uncertain == [
        derived,
        "Does opportunity cost only count the best option, or all of them?",
    ]
    assert session.learner_state.has_gap() is True

    restored = restore_session(cid)
    assert restored.reading_signals == [
        {"kind": "confused", "position": 0},
        {"kind": "clear", "position": 2},
    ]
    assert restored.stuck_points == [derived, "Does opportunity cost only count the best option, or all of them?"]
    assert "Does opportunity cost only count the best option, or all of them?" in restored.learner_state.uncertain
    assert derived in restored.learner_state.uncertain


def test_confused_signal_alone_closes_gap() -> None:
    """V0.3.0 patch 1 — tapping "🤔 didn't get it" without text still counts as an understanding gap."""
    session = LearningSession("opportunity cost", "Paragraph one\n\nParagraph two")
    cid = session.begin()
    session.record_reading_signal("confused", 1)
    assert session.learner_state.has_gap() is True
    assert 'The learner marked paragraph 2 as "didn\'t get it" while reading' in session.learner_state.uncertain
    row = database.get_concept(cid)["signals"]
    import json as _json

    payload = _json.loads(row)
    assert any(s["kind"] == "confused" for s in payload["reading_signals"])


def test_clear_and_match_signals_do_not_create_gaps() -> None:
    """V0.3.0 patch 1 — only "didn't get it" feeds the Learner State; 💡/✓ don't create gaps."""
    session = LearningSession("opportunity cost", "Paragraph one")
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
    """Confidence prediction before validation: records attempt/prediction, back-fills actual_level after judging."""
    session, _ = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", understood=["core"], quality="deep"),
    )
    cid = session.begin()
    session.start_validation()
    session.record_confidence_prediction("😊 I think I can explain it clearly")

    assert session.confidence_predictions == [
        {"attempt": 1, "prediction": "😊 I think I can explain it clearly"}
    ]
    assert session.should_ask_confidence() is False  # not asked again after recording

    session.submit_validation("answer")
    assert session.confidence_predictions[-1]["actual_level"] == "relationship"

    payload = json.loads(database.get_concept(cid)["signals"])
    assert payload["confidence"][-1]["attempt"] == 1
    assert payload["confidence"][-1]["actual_level"] == "relationship"


def test_should_ask_confidence_is_boolean() -> None:
    s1, _ = make_session()
    s1.begin()
    assert isinstance(s1.should_ask_confidence(), bool)


def test_intervention_feedback_recorded_in_history() -> None:
    """Post-intervention feedback: written to the feedback list and learner_state.intervention_history, and restorable."""
    session, _ = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", uncertain=["boundary unclear"]),
        intervention_reply("counterexample", "If you give up three options, is the opportunity cost all three added together?"),
        feedback_reply("Your answer mentioned: ..."),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("answer")
    assert session.stage == "intervention"

    session.record_intervention_feedback("unclear")
    assert session.intervention_feedback_list == [
        {"action": "counterexample", "feedback": "unclear"}
    ]
    assert (
        session.learner_state.intervention_history[-1]["feedback"] == "unclear"
    )
    assert session.feedback_pending() is False  # already given feedback, no more prompts

    row = database.get_concept(cid)
    assert json.loads(row["intervention_feedback"]) == [
        {"action": "counterexample", "feedback": "unclear"}
    ]

    restored = restore_session(cid)
    assert restored.intervention_feedback_list == [
        {"action": "counterexample", "feedback": "unclear"}
    ]
    # after restore `_current_intervention` is reset to None, so the feedback is not asked again
    assert restored.feedback_pending() is False


def test_decider_prompt_uses_past_intervention_feedback() -> None:
    """The intervention decider receives past feedback and avoids the approach the learner said was "still confusing"."""
    session, transport = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", uncertain=["boundary unclear"]),
        intervention_reply("counterexample", "Counterexample: how do you calculate three options?"),
        feedback_reply("Your answer mentioned: ..."),
        update_reply("relationship", uncertain=["boundary unclear"], next_best_action="question"),
        intervention_reply("example", "Example: the trade-off when you want to buy a computer"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("answer")
    session.record_intervention_feedback("unclear")
    session.submit_intervention_answer("my answer")

    contents = [p["messages"][1]["content"] for p in transport.requests]
    deciders = [
        c for c in contents if "minimal-intervention decider" in c and "The learner's feedback on the interventions given" in c
    ]
    assert deciders, "expected the decider to receive the intervention feedback history"
    assert "counterexample" in deciders[-1]
    assert "unclear" in deciders[-1]


# ------------------------------------------------- V0.3.0 patch 2/3 intervention loop


def test_ineffective_intervention_records_and_escalates_floor() -> None:
    """An ineffective intervention is recorded with effective=False, the streak counter +1, and the next step is hard-raised to the next priority.

    Even if the AI disobeys (still returns hint), it is forcibly raised to example (>= the lowest allowed level).
    """
    session, transport = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", uncertain=["boundary unclear"]),
        intervention_reply("hint", "Hint: what does \"cost\" refer to"),
        feedback_reply("Your answer mentioned: ..."),
        update_reply("relationship", uncertain=["boundary unclear"], next_best_action="question"),
        intervention_reply("hint", "Hint: think about the boundary again"),  # below the floor, must be raised
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("answer")
    assert session.current_intervention()["action"] == "hint"

    result = session.submit_intervention_answer("my answer")
    assert result["stage"] == "intervention"
    assert session.stage == "intervention"
    # recorded ineffective + streak
    assert session._consecutive_ineffective == 1
    assert session.learner_state.intervention_history[-2]["effective"] is False
    assert session.learner_state.intervention_history[-2]["action"] == "hint"
    # next decision is forced to escalate (AI returns hint → raised to example)
    assert session.current_intervention()["action"] == "example"

    # the decider prompt carries the "lowest allowed level" constraint
    decider_prompts = [
        p["messages"][1]["content"]
        for p in transport.requests
        if "minimal-intervention decider" in p["messages"][1]["content"]
    ]
    assert "The last intervention did not help the learner progress" in decider_prompts[-1]
    assert "example (concrete example)" in decider_prompts[-1]
    # the constraint is not lost on persistence: the intervention_history snapshot carries the effective marker
    restored = restore_session(cid, client=session.client)
    assert restored.learner_state.intervention_history[-1]["effective"] is False
    assert restored.learner_state.intervention_history[-1]["action"] == "hint"


def test_two_ineffective_interventions_finish() -> None:
    """Two consecutive ineffective interventions → complete directly, no further looping."""
    session, _ = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", uncertain=["boundary unclear"]),
        intervention_reply("hint", "Hint 1"),
        feedback_reply("Your answer mentioned: ..."),
        update_reply("relationship", uncertain=["boundary unclear"], next_best_action="question"),
        intervention_reply("question", "Question: how do you actually calculate the boundary?"),  # floor=example, question continues at a stronger level
        update_reply("relationship", uncertain=["boundary unclear"], next_best_action="question"),
    )
    cid = session.begin()
    session.start_validation()
    session.submit_validation("answer")
    session.submit_intervention_answer("answer 1")
    assert session.stage == "intervention"
    assert session._consecutive_ineffective == 1

    result = session.submit_intervention_answer("answer 2")
    assert result["stage"] == "complete"
    assert session.stage == "complete"
    assert session.phase == "connections"
    assert "final_note" in result
    assert session._consecutive_ineffective == 2
    # joins the review queue
    row = database.get_concept(cid)
    assert row["next_review_date"] == (date.today() + timedelta(days=1)).isoformat()


def test_effective_intervention_resets_counter() -> None:
    """A level gain → record effective=True and reset the consecutive-ineffective counter."""
    session, _ = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", uncertain=["boundary unclear"]),
        intervention_reply("hint", "Hint"),
        feedback_reply("Your answer mentioned: ..."),
        update_reply("application", understood=["can use it"], next_best_action="none"),
    )
    session.begin()
    session.start_validation()
    session.submit_validation("answer")

    result = session.submit_intervention_answer("I gave a life example")
    assert result["stage"] == "complete"
    assert session._consecutive_ineffective == 0
    assert (
        session.learner_state.intervention_history[-1]["effective"] is True
    )
    assert (
        session.learner_state.understanding_level == "application"
    )


def test_escalation_caps_at_top_of_ladder() -> None:
    """At the top of the ladder (question), an ineffective intervention has nothing above it → respect the AI's closing decision."""
    session, _ = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", uncertain=["boundary unclear"]),
        intervention_reply("question", "Question 1"),
        feedback_reply("Your answer mentioned: ..."),
        update_reply("relationship", uncertain=["boundary unclear"], next_best_action="question"),
        intervention_reply("none", "Let's stop here and push on in tomorrow's review.", requires_user_response=False),
    )
    session.begin()
    session.start_validation()
    session.submit_validation("answer")
    assert session.current_intervention()["action"] == "question"

    result = session.submit_intervention_answer("answer 1")
    # nothing above question → minimum_action is empty → the AI's none closing decision takes effect
    assert result["stage"] == "complete"
    assert "Let's stop here" in result["final_note"]


def test_calculate_answer_richness_by_length_and_example() -> None:
    """V0.3.1 hotfix — answer richness grading: <30 → simple; 30–80 → moderate; >80 with an example → rich."""
    session, _ = make_session()

    assert session._calculate_answer_richness("Opportunity cost") == "simple"
    assert session._calculate_answer_richness(
        "Choose A, give up B; you weigh the second-best option you lose."
    ) == "moderate"
    long_no_example = "Opportunity cost is the value of the best option you give up when choosing among multiple choices, " * 3
    assert len(long_no_example) > 80
    assert session._calculate_answer_richness(long_no_example) == "moderate"
    long_with_example = long_no_example + "like choosing to study means giving up the wages you could have earned"
    assert session._calculate_answer_richness(long_with_example) == "rich"


def test_decider_prompt_uses_last_answer_richness() -> None:
    """V0.3.1 hotfix — the intervention decider prompt carries the richness of the last answer."""
    session, transport = make_session(
        validation_task_reply("task"),
        analysis_reply("relationship", uncertain=["boundary unclear"]),
        intervention_reply("counterexample", "If you give up three options, is the opportunity cost all three added together?"),
        feedback_reply("Your answer mentioned: ..."),
    )
    session.begin()
    session.start_validation()
    session.submit_validation("Opportunity cost")
    prompt = next(
        p["messages"][1]["content"] for p in transport.requests
        if "minimal-intervention decider" in p["messages"][1]["content"]
    )
    assert "Richness" in prompt
    assert "simple" in prompt  # "Opportunity cost" <30 chars → simple
