"""Tests for prompt templates and JSON parsing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core import (
    SYSTEM_PROMPT,
    CheckAnswerResult,
    ConnectionSuggestion,
    SummaryResult,
    build_messages,
    check_answer_prompt,
    connections_prompt,
    parse_json_response,
    question_prompt,
    summary_prompt,
    validate_response,
    validate_response_list,
)


# ---------------------------------------------------------------- question

def test_question_prompt_contains_context_and_guide() -> None:
    prompt = question_prompt(1, title="机会成本", source_text="原文……")
    assert "机会成本" in prompt
    assert "原文……" in prompt
    assert "第一层追问" in prompt


@pytest.mark.parametrize(
    ("layer", "marker"),
    [(1, "第一层追问"), (2, "第二层追问"), (3, "第三层追问"), (4, "第四层追问")],
)
def test_question_prompt_all_layers(layer: int, marker: str) -> None:
    prompt = question_prompt(layer, title="X", source_text="Y")
    assert marker in prompt


def test_question_prompt_layer4_includes_related_concepts() -> None:
    prompt = question_prompt(
        4, title="机会成本", source_text="Y", related_concepts=["沉没成本", "边际效用"]
    )
    assert "沉没成本" in prompt
    assert "边际效用" in prompt


def test_question_prompt_layer4_without_related_falls_back() -> None:
    prompt = question_prompt(4, title="机会成本", source_text="Y")
    assert "你之前学过的其他概念" in prompt


def test_question_prompt_includes_qa_history_when_given() -> None:
    prompt = question_prompt(
        2, title="X", source_text="Y", qa_history="A1: 答1\nA2: 答2"
    )
    assert "回答历史" in prompt


def test_question_prompt_invalid_layer_rejected() -> None:
    with pytest.raises(ValueError):
        question_prompt(5, title="X", source_text="Y")


def test_question_prompt_is_plain_text_not_json() -> None:
    prompt = question_prompt(1, title="X", source_text="Y")
    assert '{"' not in prompt


# ------------------------------------------------------------ structured JSON

def test_check_answer_prompt_requests_json() -> None:
    prompt = check_answer_prompt(
        title="机会成本", source_text="S", question="Q?", answer="A"
    )
    assert "is_correct" in prompt
    assert '{"is_correct": true或false' in prompt


def test_summary_prompt_requests_json() -> None:
    prompt = summary_prompt(title="机会成本", user_definition="U", qa_history="H")
    assert "breakthrough" in prompt
    assert "tomorrow_hook" in prompt


def test_connections_prompt_lists_concepts() -> None:
    prompt = connections_prompt(
        title="机会成本", source_text="S", all_concepts=["沉没成本", "边际效用"]
    )
    assert "- 沉没成本" in prompt
    assert "- 边际效用" in prompt
    assert "concept_title" in prompt


def test_build_messages_has_system_and_user() -> None:
    messages = build_messages("hello")
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hello"},
    ]


# --------------------------------------------------------------- JSON parsing

def test_parse_json_response_with_fences() -> None:
    text = '```json\n{"is_correct": true, "feedback": "对"}\n```'
    assert parse_json_response(text) == {"is_correct": True, "feedback": "对"}


def test_parse_json_response_with_surrounding_text() -> None:
    text = '好的，结果如下：\n[{"concept_title": "A", "relation_text": "R"}]\n完毕'
    assert parse_json_response(text) == [
        {"concept_title": "A", "relation_text": "R"}
    ]


def test_parse_json_response_no_json_raises() -> None:
    with pytest.raises(ValueError):
        parse_json_response("没有任何 JSON")


def test_parse_json_response_invalid_json_raises() -> None:
    with pytest.raises(ValueError):
        parse_json_response('{"is_correct": true')


def test_validate_response_ok() -> None:
    result = validate_response(
        '{"is_correct": true, "feedback": "很好", "hint": null}',
        CheckAnswerResult,
    )
    assert result.is_correct is True
    assert result.feedback == "很好"
    assert result.hint is None


def test_validate_response_rejects_bad_structure() -> None:
    with pytest.raises(ValidationError):
        validate_response('{"feedback": "缺 is_correct"}', CheckAnswerResult)


def test_validate_response_list() -> None:
    suggestions = validate_response_list(
        '[{"concept_title": "沉没成本", "relation_text": "都关于选择"},'
        '{"concept_title": "边际效用", "relation_text": "都关于价值"}]',
        ConnectionSuggestion,
    )
    assert [s.concept_title for s in suggestions] == ["沉没成本", "边际效用"]


def test_validate_response_list_rejects_non_array() -> None:
    with pytest.raises(ValueError):
        validate_response_list('{"concept_title": "A"}', ConnectionSuggestion)


def test_summary_result_model() -> None:
    result = SummaryResult(
        breakthrough="我终于搞懂了机会成本",
        tomorrow_hook="边际效用和机会成本有什么关系？",
    )
    assert result.tomorrow_hook.startswith("边际效用")
