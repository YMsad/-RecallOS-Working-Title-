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
    deepening_question_prompt,
    DEEPENING_LEVEL_GUIDES,
    intervention_decider_prompt,
    learner_state_analyzer_prompt,
    learner_state_updater_prompt,
    parse_json_response,
    question_prompt,
    simplify_explanation_prompt,
    summary_prompt,
    validate_response,
    validate_response_list,
    validation_feedback_prompt,
    validation_task_prompt,
    warmup_prompt,
)


# ---------------------------------------------------------------- question

def test_question_prompt_contains_context_and_guide() -> None:
    prompt = question_prompt(1, title="opportunity cost", source_text="Original text...")
    assert "opportunity cost" in prompt
    assert "Original text..." in prompt
    assert "Layer 1" in prompt


@pytest.mark.parametrize(
    ("layer", "marker"),
    [(1, "Layer 1"), (2, "Layer 2"), (3, "Layer 3"), (4, "Layer 4")],
)
def test_question_prompt_all_layers(layer: int, marker: str) -> None:
    prompt = question_prompt(layer, title="X", source_text="Y")
    assert marker in prompt


def test_question_prompt_layer4_includes_related_concepts() -> None:
    prompt = question_prompt(
        4, title="opportunity cost", source_text="Y", related_concepts=["sunk cost", "marginal utility"]
    )
    assert "sunk cost" in prompt
    assert "marginal utility" in prompt


def test_question_prompt_layer4_without_related_falls_back() -> None:
    prompt = question_prompt(4, title="opportunity cost", source_text="Y")
    assert "other concepts you've learned before" in prompt


def test_question_prompt_includes_qa_history_when_given() -> None:
    prompt = question_prompt(
        2, title="X", source_text="Y", qa_history="A1: Answer 1\nA2: Answer 2"
    )
    assert "The learner's previous answers" in prompt


def test_question_prompt_invalid_layer_rejected() -> None:
    with pytest.raises(ValueError):
        question_prompt(5, title="X", source_text="Y")


def test_question_prompt_is_plain_text_not_json() -> None:
    prompt = question_prompt(1, title="X", source_text="Y")
    assert '{"' not in prompt


@pytest.mark.parametrize(
    ("model", "marker"),
    [
        ("golden_circle", "Golden Circle"),
        ("scenario", "Scenario-based questioning"),
        ("analogy", "Structural analogy"),
        ("first_principles", "First principles"),
    ],
)
def test_question_prompt_with_thinking_model(model: str, marker: str) -> None:
    prompt = question_prompt(1, title="X", source_text="Y", model=model)
    assert marker in prompt
    assert "Thinking model" in prompt


def test_question_prompt_invalid_model_rejected() -> None:
    with pytest.raises(ValueError):
        question_prompt(1, title="X", source_text="Y", model="nope")


def test_question_prompt_with_cognitive_contrast() -> None:
    prompt = question_prompt(
        2, title="opportunity cost", source_text="Y", cognitive_contrast=True
    )
    assert "Cognitive contrast" in prompt
    assert "misunderstand" in prompt
    plain = question_prompt(2, title="opportunity cost", source_text="Y")
    assert "Cognitive contrast" not in plain


def test_warmup_prompt_is_plain_text_with_title() -> None:
    prompt = warmup_prompt(title="opportunity cost", source_text="Original text...")
    assert "opportunity cost" in prompt
    assert "warm-up" in prompt
    assert "In one sentence" in prompt
    assert '{"' not in prompt


# ------------------------------------------------------------ structured JSON

def test_check_answer_prompt_requests_json() -> None:
    prompt = check_answer_prompt(
        title="opportunity cost", source_text="S", question="Q?", answer="A"
    )
    assert "is_correct" in prompt
    assert '{"is_correct": true or false' in prompt


def test_summary_prompt_requests_json() -> None:
    prompt = summary_prompt(title="opportunity cost", user_definition="U", qa_history="H")
    assert "breakthrough" in prompt
    assert "tomorrow_hook" in prompt


def test_connections_prompt_lists_concepts() -> None:
    prompt = connections_prompt(
        title="opportunity cost", source_text="S", all_concepts=["sunk cost", "marginal utility"]
    )
    assert "- sunk cost" in prompt
    assert "- marginal utility" in prompt
    assert "concept_title" in prompt


def test_build_messages_has_system_and_user() -> None:
    messages = build_messages("hello")
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hello"},
    ]


# --------------------------------------------------------------- JSON parsing

def test_parse_json_response_with_fences() -> None:
    text = '```json\n{"is_correct": true, "feedback": "Correct"}\n```'
    assert parse_json_response(text) == {"is_correct": True, "feedback": "Correct"}


def test_parse_json_response_with_surrounding_text() -> None:
    text = 'OK, here is the result:\n[{"concept_title": "A", "relation_text": "R"}]\nDone'
    assert parse_json_response(text) == [
        {"concept_title": "A", "relation_text": "R"}
    ]


def test_parse_json_response_no_json_raises() -> None:
    with pytest.raises(ValueError):
        parse_json_response("there is no JSON here")


def test_parse_json_response_invalid_json_raises() -> None:
    with pytest.raises(ValueError):
        parse_json_response('{"is_correct": true')


def test_validate_response_ok() -> None:
    result = validate_response(
        '{"is_correct": true, "feedback": "Great", "hint": null}',
        CheckAnswerResult,
    )
    assert result.is_correct is True
    assert result.feedback == "Great"
    assert result.hint is None


def test_validate_response_rejects_bad_structure() -> None:
    with pytest.raises(ValidationError):
        validate_response('{"feedback": "missing is_correct"}', CheckAnswerResult)


def test_validate_response_list() -> None:
    suggestions = validate_response_list(
        '[{"concept_title": "sunk cost", "relation_text": "both about choice"},'
        '{"concept_title": "marginal utility", "relation_text": "both about value"}]',
        ConnectionSuggestion,
    )
    assert [s.concept_title for s in suggestions] == ["sunk cost", "marginal utility"]


def test_validate_response_list_rejects_non_array() -> None:
    with pytest.raises(ValueError):
        validate_response_list('{"concept_title": "A"}', ConnectionSuggestion)


def test_summary_result_model() -> None:
    result = SummaryResult(
        breakthrough="I finally understood opportunity cost",
        tomorrow_hook="What is the relationship between marginal utility and opportunity cost?",
    )
    assert result.tomorrow_hook.startswith("What is the relationship")


# --------------------------------------------------------- V0.3.0 new prompts


def test_detect_text_type_prompt_requests_json() -> None:
    prompt = detect_text_type_prompt(raw_text="Opportunity cost is the second-best choice given up in a decision")
    assert "Pasted material" in prompt
    assert "text_type" in prompt
    assert "title_hint" in prompt
    assert '"reason"' in prompt


def test_detect_text_type_result_validates() -> None:
    result = validate_response(
        '{"text_type": "concept", "title_hint": "opportunity cost", "reason": null}',
        TextTypeResult,
    )
    assert result.text_type == "concept"
    assert result.title_hint == "opportunity cost"
    assert result.reason is None


def test_validation_task_prompt_requests_json() -> None:
    prompt = validation_task_prompt(source_text="S", concept="opportunity cost")
    assert "validation task" in prompt or "validation" in prompt
    assert '"task"' in prompt
    assert '"type"' in prompt
    assert '"difficulty"' in prompt
    assert "opportunity cost" in prompt


def test_validation_task_prompt_uses_text_type() -> None:
    prompt = validation_task_prompt(source_text="S", concept="opportunity cost", text_type="concept")
    assert "concept" in prompt
    assert "text_type" in prompt


def test_validation_task_validates() -> None:
    result = validate_response(
        '{"task": "explain in your own words", "type": "summary", "difficulty": 2}',
        ValidationTask,
    )
    assert result.task == "explain in your own words"
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
        title="opportunity cost", task="explain in one sentence", target="key points", answer="Answer"
    )
    assert "validation task" in prompt
    assert "is_correct" in prompt
    assert '"missing"' in prompt


def test_legacy_validation_task_prompt_requests_json() -> None:
    from core import _legacy_validation_task_prompt

    prompt = _legacy_validation_task_prompt(title="opportunity cost", source_text="S", qa_history="H")
    assert "validation task" in prompt
    assert '"task"' in prompt
    assert '"target"' in prompt


def test_learner_state_analyzer_prompt_requests_json() -> None:
    prompt = learner_state_analyzer_prompt(
        source_text="S", concept="opportunity cost", task="T", user_answer="A"
    )
    assert "learner-state analyzer" in prompt
    assert '"understanding_level"' in prompt
    assert '"understood"' in prompt
    assert '"uncertain"' in prompt
    assert '"misconceptions"' in prompt
    assert '"last_response_quality"' in prompt


def test_learner_state_analyzer_prompt_includes_context_when_given() -> None:
    prompt = learner_state_analyzer_prompt(
        source_text="S", concept="opportunity cost", task="T", user_answer="A", context="previous conversation"
    )
    assert "Previous conversation" in prompt


def test_analyzer_prompt_includes_user_signals() -> None:
    """V0.3.1 — the analyzer carries the learning goal / stuck points / confidence prediction."""
    prompt = learner_state_analyzer_prompt(
        source_text="S",
        concept="opportunity cost",
        task="T",
        user_answer="A",
        learning_goal="apply",
        stuck_points="does opportunity cost only count the best option?",
        confidence_prediction="😊 I think I can explain it clearly",
    )
    assert "apply it in practice" in prompt  # apply goal hint
    assert "does opportunity cost only count the best option?" in prompt
    assert "😊 I think I can explain it clearly" in prompt

    plain = learner_state_analyzer_prompt(
        source_text="S", concept="X", task="T", user_answer="A"
    )
    assert "apply it in practice" not in plain  # default understand, no apply goal
    assert "Goal: understand the concept itself" in plain


def test_validation_feedback_prompt_is_concrete_and_contrastive() -> None:
    """V0.3.1 hotfix — feedback must be specific, comparative and actionable, not a bare "not quite there"."""
    prompt = validation_feedback_prompt(
        concept_title="opportunity cost",
        user_answer="It helps with decision-making",
        task_description="Explain opportunity cost in your own words",
        reading_answers=[
            {"paragraph_index": 0, "answer": "Opportunity cost is the option you give up"}
        ],
    )
    assert "Socratic mentor" in prompt
    assert "opportunity cost" in prompt
    assert "It helps with decision-making" in prompt
    assert "Opportunity cost is the option you give up" in prompt  # reading answers in context
    # structural requirements: compare / example / why
    assert "Your answer mentioned" in prompt
    assert "But the core of opportunity cost is" in prompt
    assert "Compare:" in prompt
    assert "Try this example:" in prompt
    assert "summarize how well they understand" in prompt
    assert 'Never just say "not quite there"' in prompt


def test_validation_feedback_prompt_optional_reading_answers() -> None:
    prompt = validation_feedback_prompt(
        concept_title="compound interest",
        user_answer="interest on interest",
        task_description="Explain compound interest",
    )
    assert "The learner's answer" in prompt and "interest on interest" in prompt
    assert "The learner's understanding while reading" not in prompt  # omitted without reading answers


def test_intervention_decider_prompt_includes_feedback_history() -> None:
    """V0.3.1 — the decider receives feedback on past interventions."""
    prompt = intervention_decider_prompt(
        source_text="S",
        concept="opportunity cost",
        learner_state="{}",
        intervention_history='[{"action": "counterexample", "feedback": "unclear"}]',
    )
    assert "The learner's feedback on the interventions given" in prompt
    assert '"action": "counterexample"' in prompt
    assert "unclear" in prompt

    plain = intervention_decider_prompt(
        source_text="S", concept="X", learner_state="{}"
    )
    assert "The learner's feedback on the interventions given" not in plain


def test_intervention_decider_forces_intensity_ladder() -> None:
    """V0.3.0 patch 3 — the decider prompt embeds a forced low-to-high intervention ladder."""
    prompt = intervention_decider_prompt(
        source_text="S", concept="opportunity cost", learner_state="{}"
    )
    assert "hint (one-line hint) → example (concrete example) → analogy (analogy) → counterexample (counterexample) → question (direct question)" in prompt
    assert "Only escalate to a heavier intervention once the lighter one has failed." in prompt
    assert "The last intervention did not help the learner progress" not in prompt  # no constraint without a floor

    escalated = intervention_decider_prompt(
        source_text="S",
        concept="opportunity cost",
        learner_state="{}",
        minimum_action="example",
    )
    assert "The last intervention did not help the learner progress" in escalated
    assert "example (concrete example)" in escalated
    assert "start from this level" in escalated


def test_intervention_decider_grades_question_by_answer_richness() -> None:
    """V0.3.1 hotfix — the decider receives the answer richness and must push at the matching level."""
    prompt = intervention_decider_prompt(
        source_text="S",
        concept="opportunity cost",
        learner_state="{}",
        answer_richness="rich",
    )
    assert "Richness" in prompt
    assert "rich" in prompt
    assert "Counterfactual" in prompt  # rich → counterfactual "what if it didn't exist?"
    assert "what would happen if this concept didn't exist" in prompt

    simple = intervention_decider_prompt(
        source_text="S", concept="opportunity cost", learner_state="{}", answer_richness="simple"
    )
    assert "Everyday analogy" in simple
    assert "what is this like?" in simple

    plain = intervention_decider_prompt(
        source_text="S", concept="X", learner_state="{}"
    )
    assert "Richness" not in plain  # not provided → doesn't distort the default decision


def test_deepening_question_prompt_grades_by_level() -> None:
    """V0.3.1 hotfix — deepening_question_prompt templates a follow-up by richness level."""
    simple = deepening_question_prompt(level="simple", concept="opportunity cost", source_text="Original text")
    assert "what is this like?" in simple
    assert "Everyday analogy" in simple

    moderate = deepening_question_prompt(level="moderate", concept="opportunity cost", source_text="Original text")
    assert "what does this relate to?" in moderate
    assert DEEPENING_LEVEL_GUIDES["moderate"] in moderate

    rich = deepening_question_prompt(
        level="rich", concept="opportunity cost", source_text="Original text", qa_history="Q1→A1"
    )
    assert "what would happen if this concept didn't exist" in rich
    assert "Q1→A1" in rich  # answer history in context

    with pytest.raises(ValueError):
        deepening_question_prompt(level="unknown", concept="X", source_text="Y")


def test_learner_state_analysis_validates() -> None:
    result = validate_response(
        '{"understanding_level": "relationship", "understood": ["understands the relationship"], '
        '"uncertain": [], "misconceptions": [], "last_response_quality": "partial"}',
        LearnerStateAnalysis,
    )
    assert result.understanding_level == "relationship"
    assert result.understood == ["understands the relationship"]


def test_intervention_decider_prompt_requests_json() -> None:
    prompt = intervention_decider_prompt(
        source_text="S", concept="opportunity cost", learner_state="{}"
    )
    assert "minimal-intervention decider" in prompt
    assert '"action"' in prompt
    assert '"content"' in prompt
    assert '"requires_user_response"' in prompt
    assert "Current stage is validation" in prompt
    deepening = intervention_decider_prompt(
        source_text="S", concept="opportunity cost", learner_state="{}", mode="deepening"
    )
    assert "Current stage is deepening" in deepening


def test_intervention_validates() -> None:
    result = validate_response(
        '{"action": "counterexample", "reason": "boundary unclear", '
        '"content": "If you give up three options, is the opportunity cost all three added together?", '
        '"requires_user_response": true}',
        Intervention,
    )
    assert result.action == "counterexample"
    assert result.content.startswith("If you give up")


def test_learner_state_updater_prompt_requests_json() -> None:
    prompt = learner_state_updater_prompt(
        source_text="S", concept="opportunity cost", intervention="hint",
        user_answer="A", learner_state="{}"
    )
    assert "learner-state updater" in prompt
    assert '"next_best_action"' in prompt
    assert '"understanding_level"' in prompt


def test_learner_state_updater_prompt_includes_goal() -> None:
    """V0.3.1 — the updater carries the learning goal as the bar for passing."""
    prompt = learner_state_updater_prompt(
        source_text="S", concept="opportunity cost", intervention="hint",
        user_answer="A", learner_state="{}", learning_goal="exam",
    )
    assert "master it for an exam" in prompt
    assert "accurate terms" in prompt
    default = learner_state_updater_prompt(
        source_text="S", concept="opportunity cost", intervention="hint",
        user_answer="A", learner_state="{}",
    )
    assert "master it for an exam" not in default


def test_learner_state_update_validates() -> None:
    result = validate_response(
        '{"understanding_level": "application", "understood": ["can use it"], '
        '"uncertain": [], "misconceptions": [], "last_response_quality": "deep", '
        '"next_best_action": "none"}',
        LearnerStateUpdate,
    )
    assert result.understanding_level == "application"
    assert result.next_best_action == "none"


def test_deepening_offer_prompt_requests_json() -> None:
    prompt = deepening_offer_prompt(concept="opportunity cost", understanding_level="relationship")
    assert "learning coach" in prompt
    assert "opportunity cost" in prompt
    assert '"offer"' in prompt
    assert '"options"' in prompt


def test_deepening_offer_validates() -> None:
    result = validate_response(
        '{"offer": "You\'ve got the core — want to go one layer deeper?", "options": ["Go deeper", "That\'s enough"]}',
        DeepeningOffer,
    )
    assert result.offer == "You've got the core — want to go one layer deeper?"
    assert result.options == ["Go deeper", "That's enough"]


def test_simplify_explanation_prompt_is_plain_text() -> None:
    prompt = simplify_explanation_prompt(
        title="opportunity cost", source_text="S", explanation="the old explanation was a bit convoluted"
    )
    assert "not simple enough" in prompt or "another notch" in prompt
    assert "a bit convoluted" in prompt
    assert '{"' not in prompt


@pytest.mark.parametrize(
    ("qtype", "marker"),
    [
        ("verification_plus", "Re-verify"),
        ("connection", "Connection"),
        ("counterfactual", "Counterfactual"),
        ("action", "Action"),
        ("first_principles", "First principles"),
    ],
)
def test_legacy_deeper_question_prompt_all_types(qtype: str, marker: str) -> None:
    prompt = _legacy_deeper_question_prompt(
        title="opportunity cost", source_text="S", question_type=qtype
    )
    assert marker in prompt
    assert "Deeper probe" in prompt
    assert '{"' not in prompt


def test_legacy_deeper_question_prompt_includes_qa_history_when_given() -> None:
    prompt = _legacy_deeper_question_prompt(
        title="opportunity cost",
        source_text="S",
        question_type="connection",
        qa_history="answer records",
    )
    assert "answer records" in prompt


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
