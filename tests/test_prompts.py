"""Tests for prompt templates and JSON parsing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core import (
    DEEPER_QUESTION_ORDER,
    SYSTEM_PROMPT,
    CheckAnswerResult,
    ConnectionSuggestion,
    DeepeningOffer,
    Intervention,
    LearnerStateAnalysis,
    LearnerStateUpdate,
    SummaryResult,
    TextTypeResult,
    ValidationTask,
    _legacy_deeper_question_prompt,
    _legacy_validate_answer_prompt,
    build_messages,
    check_answer_prompt,
    connections_prompt,
    deepening_offer_prompt,
    detect_text_type_prompt,
    intervention_decider_prompt,
    learner_state_analyzer_prompt,
    learner_state_updater_prompt,
    parse_json_response,
    question_prompt,
    simplify_explanation_prompt,
    summary_prompt,
    validate_response,
    validate_response_list,
    validation_task_prompt,
    warmup_prompt,
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


@pytest.mark.parametrize(
    ("model", "marker"),
    [
        ("golden_circle", "黄金圈法则"),
        ("scenario", "场景化提问"),
        ("analogy", "结构性类比"),
        ("first_principles", "第一性原理"),
    ],
)
def test_question_prompt_with_thinking_model(model: str, marker: str) -> None:
    prompt = question_prompt(1, title="X", source_text="Y", model=model)
    assert marker in prompt
    assert "思维模型" in prompt


def test_question_prompt_invalid_model_rejected() -> None:
    with pytest.raises(ValueError):
        question_prompt(1, title="X", source_text="Y", model="nope")


def test_question_prompt_with_cognitive_contrast() -> None:
    prompt = question_prompt(
        2, title="机会成本", source_text="Y", cognitive_contrast=True
    )
    assert "认知反差" in prompt
    assert "误解" in prompt
    plain = question_prompt(2, title="机会成本", source_text="Y")
    assert "认知反差" not in plain


def test_warmup_prompt_is_plain_text_with_title() -> None:
    prompt = warmup_prompt(title="机会成本", source_text="原文……")
    assert "机会成本" in prompt
    assert "预热" in prompt
    assert "用一句话说" in prompt
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


# --------------------------------------------------------- V0.3.0 new prompts


def test_detect_text_type_prompt_requests_json() -> None:
    prompt = detect_text_type_prompt(raw_text="机会成本是指在决策中放弃的次优选择")
    assert "粘贴的材料原文" in prompt
    assert "text_type" in prompt
    assert "title_hint" in prompt
    assert '"reason"' in prompt


def test_detect_text_type_result_validates() -> None:
    result = validate_response(
        '{"text_type": "concept", "title_hint": "机会成本", "reason": null}',
        TextTypeResult,
    )
    assert result.text_type == "concept"
    assert result.title_hint == "机会成本"
    assert result.reason is None


def test_validation_task_prompt_requests_json() -> None:
    prompt = validation_task_prompt(source_text="S", concept="机会成本")
    assert "验证任务" in prompt or "验证" in prompt
    assert '"task"' in prompt
    assert '"type"' in prompt
    assert '"difficulty"' in prompt
    assert "机会成本" in prompt


def test_validation_task_prompt_uses_text_type() -> None:
    prompt = validation_task_prompt(source_text="S", concept="机会成本", text_type="concept")
    assert "concept" in prompt
    assert "text_type" in prompt


def test_validation_task_validates() -> None:
    result = validate_response(
        '{"task": "用自己的话解释", "type": "summary", "difficulty": 2}',
        ValidationTask,
    )
    assert result.task == "用自己的话解释"
    assert result.type == "summary"
    assert result.difficulty == 2


def test_validation_task_difficulty_bounds() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        validate_response(
            '{"task": "T", "type": "summary", "difficulty": 5}', ValidationTask
        )


def test_legacy_validate_answer_prompt_requests_json() -> None:
    prompt = _legacy_validate_answer_prompt(
        title="机会成本", task="用一句话解释", target="关键点", answer="回答"
    )
    assert "验证任务" in prompt
    assert "is_correct" in prompt
    assert '"missing"' in prompt


def test_legacy_validation_task_prompt_requests_json() -> None:
    from core import _legacy_validation_task_prompt

    prompt = _legacy_validation_task_prompt(title="机会成本", source_text="S", qa_history="H")
    assert "验证任务" in prompt
    assert '"task"' in prompt
    assert '"target"' in prompt


def test_learner_state_analyzer_prompt_requests_json() -> None:
    prompt = learner_state_analyzer_prompt(
        source_text="S", concept="机会成本", task="T", user_answer="A"
    )
    assert "学习者状态分析器" in prompt
    assert '"understanding_level"' in prompt
    assert '"understood"' in prompt
    assert '"uncertain"' in prompt
    assert '"misconceptions"' in prompt
    assert '"last_response_quality"' in prompt


def test_learner_state_analyzer_prompt_includes_context_when_given() -> None:
    prompt = learner_state_analyzer_prompt(
        source_text="S", concept="机会成本", task="T", user_answer="A", context="此前对话"
    )
    assert "此前对话" in prompt


def test_learner_state_analysis_validates() -> None:
    result = validate_response(
        '{"understanding_level": "relationship", "understood": ["理解了关系"], '
        '"uncertain": [], "misconceptions": [], "last_response_quality": "partial"}',
        LearnerStateAnalysis,
    )
    assert result.understanding_level == "relationship"
    assert result.understood == ["理解了关系"]


def test_intervention_decider_prompt_requests_json() -> None:
    prompt = intervention_decider_prompt(
        source_text="S", concept="机会成本", learner_state="{}"
    )
    assert "最小干预决策器" in prompt
    assert '"action"' in prompt
    assert '"content"' in prompt
    assert '"requires_user_response"' in prompt
    assert "验证阶段" in prompt
    deepening = intervention_decider_prompt(
        source_text="S", concept="机会成本", learner_state="{}", mode="deepening"
    )
    assert "深入阶段" in deepening


def test_intervention_validates() -> None:
    result = validate_response(
        '{"action": "counterexample", "reason": "边界不清", '
        '"content": "如果放弃三个选择，机会成本是三个加起来吗？", '
        '"requires_user_response": true}',
        Intervention,
    )
    assert result.action == "counterexample"
    assert result.content.startswith("如果放弃")


def test_learner_state_updater_prompt_requests_json() -> None:
    prompt = learner_state_updater_prompt(
        source_text="S", concept="机会成本", intervention="提示",
        user_answer="A", learner_state="{}"
    )
    assert "学习状态更新器" in prompt
    assert '"next_best_action"' in prompt
    assert '"understanding_level"' in prompt


def test_learner_state_update_validates() -> None:
    result = validate_response(
        '{"understanding_level": "application", "understood": ["能用"], '
        '"uncertain": [], "misconceptions": [], "last_response_quality": "deep", '
        '"next_best_action": "none"}',
        LearnerStateUpdate,
    )
    assert result.understanding_level == "application"
    assert result.next_best_action == "none"


def test_deepening_offer_prompt_requests_json() -> None:
    prompt = deepening_offer_prompt(concept="机会成本", understanding_level="relationship")
    assert "学习教练" in prompt
    assert "机会成本" in prompt
    assert '"offer"' in prompt
    assert '"options"' in prompt


def test_deepening_offer_validates() -> None:
    result = validate_response(
        '{"offer": "你已经抓住核心，要不要再挖一层？", "options": ["深入", "先到这里"]}',
        DeepeningOffer,
    )
    assert result.offer == "你已经抓住核心，要不要再挖一层？"
    assert result.options == ["深入", "先到这里"]


def test_simplify_explanation_prompt_is_plain_text() -> None:
    prompt = simplify_explanation_prompt(
        title="机会成本", source_text="S", explanation="旧解释有点绕"
    )
    assert "更简单" in prompt or "降一个台阶" in prompt
    assert "旧解释有点绕" in prompt
    assert '{"' not in prompt


@pytest.mark.parametrize(
    ("qtype", "marker"),
    [
        ("verification_plus", "再验证"),
        ("connection", "联系"),
        ("counterfactual", "反事实"),
        ("action", "行动"),
        ("first_principles", "第一性原理"),
    ],
)
def test_legacy_deeper_question_prompt_all_types(qtype: str, marker: str) -> None:
    prompt = _legacy_deeper_question_prompt(
        title="机会成本", source_text="S", question_type=qtype
    )
    assert marker in prompt
    assert "深化" in prompt
    assert '{"' not in prompt


def test_legacy_deeper_question_prompt_includes_qa_history_when_given() -> None:
    prompt = _legacy_deeper_question_prompt(
        title="机会成本",
        source_text="S",
        question_type="connection",
        qa_history="回答记录",
    )
    assert "回答记录" in prompt


def test_legacy_deeper_question_prompt_invalid_type_rejected() -> None:
    with pytest.raises(ValueError):
        _legacy_deeper_question_prompt(title="X", source_text="Y", question_type="nope")


def test_deeper_question_order_matches_spec() -> None:
    assert DEEPER_QUESTION_ORDER == (
        "verification_plus",
        "connection",
        "counterfactual",
        "action",
        "first_principles",
    )
