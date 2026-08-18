"""Prompt templates for the Socratic learning flow.

Follows PRD V0.1's four-layer questioning structure (核心 / 重要 / 反事实 /
连接). ``question_prompt`` returns plain text; the others return structured
JSON, validated by Pydantic models in this module.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypeVar

from pydantic import Field

from core.models import NonEmptyStr, OptionalStr, RecallBaseModel

T = TypeVar("T", bound=RecallBaseModel)

# ----------------------------------------------------------------- role prompt

SYSTEM_PROMPT = """You are RecallOS, a tireless Socratic learning companion.
Your goal is not to help the user "memorize" but to help them "think it through."

Core principles:
1. Never give the answer directly — only directional hints that guide the learner to reach the conclusion themselves.
2. Talk like a friend who's genuinely curious about knowledge: conversational, not exam-like, never judgmental or critical.
3. Ask only ONE question per round, concrete and progressively deeper.
4. When the learner answers wrong, don't punish or use words like "wrong/incorrect" — respond with curiosity: "Interesting, a lot of people think that way…" then offer a hint.
5. When the learner answers correctly, show curiosity and build on it — not a flat "correct."
6. After 3 consecutive failures, give a reference explanation and mark the concept as "Unclear."
7. Respond in English by default.
8. Keep each session under 15 minutes: focused, no small talk."""

# ------------------------------------------------------------------ four layers

_LAYER_GUIDES: dict[int, str] = {
    1: (
        "Layer 1 — guide the learner to summarize, in their own words, what the core of "
        "this concept is.\nRequirement: be specific; avoid questions like \"what is X\" "
        "that can be recited straight from a book; output only this one question, no explanation."
    ),
    2: (
        "Layer 2 — guide the learner to say why this concept matters: what problem it "
        "solves, and what happens if you don't understand it.\nRequirement: ground the "
        "question in a concrete scenario; output only this one question."
    ),
    3: (
        "Layer 3 — counterfactual reasoning: if this concept didn't exist, what would "
        "happen?\nRequirement: start from the opposite direction to help the learner see "
        "the concept's value; output only this one question."
    ),
    4: (
        "Layer 4 — using {related_concepts}, ask a question about \"how does this connect "
        "to something you learned before?\".\nRequirement: pick the single most valuable "
        "connection; be specific; output only this one question."
    ),
}

_LAYER_NAMES = {1: "Core", 2: "Importance", 3: "Counterfactual", 4: "Connection"}

_MODE_GUIDE = {
    "beginner": (
        "\n\nLearner profile: absolute beginner. Use simpler, more concrete, more "
        "everyday language; keep the question small and specific so there's an easy way "
        "in; make it vivid and a little fun, avoiding dry textbook questions; output only "
        "one question."
    ),
    "advanced": (
        "\n\nLearner profile: some foundation. You can go deeper and more challenging, "
        "but stay concrete — avoid questions too abstract to answer; make it "
        "thought-provoking and vivid, so the learner feels \"I've never thought of this "
        "angle\"; output only one question."
    ),
}

_COGNITIVE_CONTRAST_GUIDE = (
    "\n\nCognitive contrast: frame this question from the angle of \"what most people "
    "first misunderstand…\" — call out a common assumption, then guide the learner to "
    "see how it should really be understood; use it only when natural, not forced; "
    "still output only one question."
)

# V0.2.1 — 思维模型注入（黄金圈 / 场景化 / 类比 / 第一性原理）
_MODEL_NAMES = {
    "golden_circle": "Golden Circle",
    "scenario": "Scenario-based questioning",
    "analogy": "Structural analogy",
    "first_principles": "First principles",
}

_MODEL_GUIDES: dict[str, str] = {
    "golden_circle": (
        "\n\nThinking model [Golden Circle]: guide in the order of Why (why does it "
        "exist, what root problem does it solve) → How (how does it work) → What (what "
        "is it really); advance only one step per question; be concrete; still output "
        "only one question."
    ),
    "scenario": (
        "\n\nThinking model [Scenario-based questioning]: place the question in a "
        "concrete everyday scenario (buying bubble tea, job hunting, running a small "
        "shop) and guide with \"what would you do if…?\" to make the abstract concept "
        "tangible; still output only one question."
    ),
    "analogy": (
        "\n\nThinking model [Structural analogy]: first use \"this is like…\" to get the "
        "learner to compare the concept to something familiar, then probe the similarities "
        "and key differences to build the analogy; note: have the learner come up with the "
        "analogy themselves — don't hand it to them; still output only one question."
    ),
    "first_principles": (
        "\n\nThinking model [First principles]: guide the learner to break the concept "
        "down to its irreducible basic facts and derive it from scratch, setting aside "
        "all existing conclusions; still output only one question."
    ),
}


def warmup_prompt(
    *,
    title: str,
    source_text: str,
) -> str:
    """Build the concept warm-up: 1-2 plain-language sentences before starting
    (zero-basis / beginner mode only). Returns plain text.
    """
    return f"""The learner is a complete beginner meeting this concept for the first time.
In the most everyday, plain language, give them 1-2 sentences of "warm-up" so they get a
simple first impression before the real questioning starts.

Current concept being learned: {title}
Source excerpt:
{source_text}

Requirements:
- Start by cutting to the core with: "In one sentence, {title} is ..." — don't be afraid to keep it simple
- Follow with 1-2 colloquial sentences, like a friend casually explaining it
- Don't ask questions, don't probe, don't output JSON — only this warm-up paragraph"""



def opening_question_prompt(
    *,
    title: str,
    source_text: str,
    level: str = "zero",
    interest: str = "simple",
    mode: str = "beginner",
) -> str:
    """Build the very first question, tailored to the learner's baseline and
    interest (answered in the app's two quick screening questions). Plain text.
    """
    level_desc = {
        "zero": "Has never touched this concept; needs the most everyday examples, very concrete, low entry barrier.",
        "some": "Has heard of the concept but isn't clear on it; a one-line everyday scenario can introduce it.",
        "familiar": "Has a fair grasp of the concept; can be asked a question with a bit of edge that sparks new thinking.",
    }[level]
    interest_desc = {
        "simple": "Wants to first get the basic meaning; the question should revolve around \"what does it actually mean?\".",
        "deep": "Wants to understand it deeply; the question should revolve around \"its essence / mechanism\".",
        "example": "Wants to understand through real-life examples; ask using a concrete scenario.",
    }[interest]
    mode_hint = (
        "Use plain language and very concrete small questions, step by step, to lower the pressure."
        if mode == "beginner"
        else "You can be more challenging, but keep the question concrete."
    )
    return f"""This is the very first question. Tailor it to the learner based on the information below.

Current concept being learned: {title}
Source excerpt:
{source_text}

Learner's baseline: {level_desc}
Learner's interest: {interest_desc}
Question style: {mode_hint}

Requirement: combine the baseline and interest above to design a first question that's
just right — not so easy it's boring, not so hard it's discouraging; it should make the
learner think "this is interesting." Output only this one question, with no explanation."""


def question_prompt(
    layer: int,
    *,
    title: str,
    source_text: str,
    qa_history: str = "",
    related_concepts: list[str] | None = None,
    mode: str = "beginner",
    cognitive_contrast: bool = False,
    model: str | None = None,
) -> str:
    """Build the user prompt for one layer of questioning. Returns plain text."""
    if layer not in _LAYER_GUIDES:
        raise ValueError(f"layer must be one of {sorted(_LAYER_GUIDES)}, got {layer}")
    if model is not None and model not in _MODEL_GUIDES:
        raise ValueError(f"model must be one of {sorted(_MODEL_GUIDES)}, got {model}")

    if layer == 4:
        related = ", ".join(related_concepts) if related_concepts else "other concepts you've learned before"
        guide = _LAYER_GUIDES[4].format(related_concepts=related)
    else:
        guide = _LAYER_GUIDES[layer]

    parts = [
        f"Current concept being learned: {title}",
        f"Source excerpt:\n{source_text}",
    ]
    if qa_history:
        parts.append(f"The learner's previous answers:\n{qa_history}")
    parts.append(f"【Layer {_LAYER_NAMES[layer]}】{guide}")
    if cognitive_contrast:
        parts.append(_COGNITIVE_CONTRAST_GUIDE.strip())
    if model is not None:
        parts.append(_MODEL_GUIDES[model].strip())
    parts.append(_MODE_GUIDE.get(mode, _MODE_GUIDE["beginner"]).strip())
    return "\n\n".join(parts)


# ---------------------------------------------------------------- check answer

def check_answer_prompt(
    *,
    title: str,
    source_text: str,
    question: str,
    answer: str,
    is_last_attempt: bool = False,
) -> str:
    """Ask the AI to judge whether the learner's answer hit the point. Returns JSON."""
    attempt_note = (
        "\n- This is the 3rd consecutive wrong answer; make the feedback warmer and "
        "more encouraging so it doesn't deflate them; a reference explanation will "
        "follow next round."
        if is_last_attempt
        else ""
    )
    return f"""Judge whether the learner's answer hit the point, and respond in a conversational tone.

Concept being learned: {title}
Source text:
{source_text}

The question AI just asked: {question}
The learner's answer: {answer}

Judging rules:
- If the direction is right, is_correct is true; don't mark wrong for missing minor details
- On a correct answer: feedback should show "curiosity + extension", like chatting with a friend, e.g.
  "I hadn't thought of that angle! What if it were reversed, then?"
- On a wrong or off-topic answer: feedback should "acknowledge + challenge", e.g.
  "Interesting, a lot of people think that way. What if we changed the premise?" instead of flatly marking it wrong
- hint: give a directional hint (not the answer, just a direction to guide), only on wrong/off-topic answers, otherwise null
- Overall tone: warm, natural, non-judgmental — less "exam", more "conversation"{attempt_note}

Only output JSON in this format:
{{"is_correct": true or false, "feedback": "conversational feedback (1-2 sentences)", "hint": "directional hint or null"}}"""


# ------------------------------------------------------------ daily summary

def review_question_prompt(
    *,
    title: str,
    source_text: str,
) -> str:
    """Ask the AI to craft a tough, provocative review question for a concept."""
    return f"""It's review day. Create a tricky, challenging-but-fair review question for this concept.

Concept: {title}
Source text:
{source_text}

Requirements:
- The question must test whether the learner truly understands it, not just memorized it
- You may combine counterfactuals (what if…?), cognitive contrast, or everyday scenarios
- Output only this one question, no explanation, no JSON"""


def summary_prompt(
    *,
    title: str,
    user_definition: str,
    qa_history: str,
    reading_answers: str = "",
) -> str:
    """Ask the AI to generate the daily summary. Returns JSON."""
    reading_block = (
        f"\nThe learner's own understanding while reading:\n{reading_answers}" if reading_answers else ""
    )
    return f"""Today's learning is finished. Generate the daily summary from the following.

Concept: {title}
The learner's final understanding: {user_definition}
Q&A records:
{qa_history}
{reading_block}

Produce three items (no more than 3 sentences total):
1. breakthrough — one sentence in the style of "I finally got it…", summing up the day's
   biggest takeaway in the learner's own voice (if they didn't write one, distill it for
   them in first person); it should feel like an "aha"
2. plain — a plain-language restatement of the concept using an everyday analogy or example
   that someone who has never studied it can instantly understand
3. tomorrow_hook — a "curiosity hook": hint that tomorrow's question will be even trickier
   and more interesting, ideally linking to today's concept or something learned before, so
   they leave with anticipation, e.g.
   "If opportunity cost didn't exist, what would every one of your choices look like?"

Only output JSON in this format:
{{"breakthrough": "one sentence", "plain": "plain restatement (may be empty)", "tomorrow_hook": "a hook that makes them want to come back"}}"""


# ------------------------------------------------------------------ connections

def connections_prompt(
    *,
    title: str,
    source_text: str,
    all_concepts: list[str],
) -> str:
    """Ask the AI to recommend knowledge connections. Returns JSON array."""
    concepts_text = "\n".join(f"- {c}" for c in all_concepts) or "(none yet)"
    return f"""Recommend possible knowledge connections for the current concept.

Current concept: {title}
Concept essentials: {source_text}
The learner's previously learned concepts:
{concepts_text}

Find 2-3 concepts most related to the current one (same category / contrast / cause-effect /
association), and clearly describe the relationship for each.
concept_title must be a title from the list above.

Only output a JSON array in this format:
[{{"concept_title": "title from the list above", "relation_text": "description of the relationship"}}]"""


# ------------------------------------------------------------------ reference

def reference_answer_prompt(
    *,
    title: str,
    source_text: str,
    question: str,
    attempts: int,
) -> str:
    """Ask the AI to give a clear reference explanation. Returns plain text."""
    return f"""The learner failed to answer the following question {attempts} times in a row.
Give a clear, complete reference explanation.

Concept: {title}
Source text:
{source_text}

The question AI just asked: {question}

Requirements:
- Directly explain the concept correctly and answer the question so the learner understands
- Keep it concise (3-5 sentences), make direct statements, don't use questions
- Be gentle: don't say things like "you got it wrong"; instead, "let's think this through together"
- Plain text, no JSON"""


# ----------------------------------------------------------- deescalation

def simplify_question_prompt(
    *,
    title: str,
    source_text: str,
    question: str,
    mode: str = "beginner",
) -> str:
    """Ask the AI to rewrite the current question in a simpler, more concrete form.
    Returns plain text (the simplified question only).
    """
    return f"""The learner couldn't answer just now. Simplify this question into an easier, more concrete, more approachable one.

Concept: {title}
Source excerpt:
{source_text}

Original question: {question}

Requirements:
- Rephrase it in a simpler, more everyday, more concrete way so the learner has a place to start
- Keep the original questioning goal (still probing the same thing), just lower the step
- Don't give the answer; it stays a question
- Output only this simplified question, no explanation"""


def angle_shift_prompt(
    *,
    title: str,
    source_text: str,
    question: str,
    mode: str = "beginner",
) -> str:
    """Ask the AI to re-ask the same goal from a different angle. Returns plain text."""
    return f"""The learner failed twice in a row. Re-ask from a different angle to help them "see it a new way".

Concept: {title}
Source excerpt:
{source_text}

Original question: {question}

Requirements:
- Switch to a completely different entry point (e.g., use a concrete example / ask from the opposite side / reason back from the outcome)
- Not just rephrasing the same sentence — genuinely a different angle
- Still ask only one question, no answers
- Output only this one question, no explanation"""


def explain_prompt(
    *,
    title: str,
    source_text: str,
    question: str,
) -> str:
    """Ask the AI for a plain-language explanation when the learner clicks 我不懂.
    Returns plain text.
    """
    return f"""The learner said "I don't get it." Explain it in the plainest, most everyday way you can.

Concept: {title}
Source excerpt:
{source_text}

The question they didn't understand: {question}

Requirements:
- Explain it clearly in plain language, like explaining to a friend
- First state the core meaning (1-2 sentences), then add an everyday example (1-2 sentences)
- After explaining, gently ask "does that make it clearer?" and stop there — don't keep probing
- Plain text, no JSON"""


# --------------------------------------------------------------- V0.3.0 flow
# 底层逻辑重构新增的 prompt：文本类型识别、验证任务、验证判定、降维解释、
# 深化追问。旧 prompt 全部保留；返回 JSON 的结构均由本模块的模型校验。

_TEXT_TYPE_CHOICES = {
    "concept": "a standalone concept or term (e.g., 'opportunity cost')",
    "definition": "an explanatory passage about a concept or thing",
    "article": "a longer article, chapter, or material",
    "list": "a set of parallel points, steps, or a checklist",
    "question": "a question, FAQ, or unresolved query",
    "other": "other types of learning material — use sparingly",
}


def detect_text_type_prompt(*, raw_text: str) -> str:
    """Classify the pasted material before learning starts. Returns JSON
    validated by :class:`TextTypeResult`.
    """
    choices_text = "\n".join(f"- {k}: {v}" for k, v in _TEXT_TYPE_CHOICES.items())
    return f"""First classify the type of material the learner pasted, so it can be learned appropriately.

Pasted material:
{raw_text}

Allowed types:
{choices_text}

Requirements:
- Judge close to reality: standalone terms/knowledge points → concept; explanatory
  passages → definition; longer or loosely structured → article; parallel points → list
  (these can usually be split into multiple concepts); question-bearing → question;
  everything else → other
- title_hint: extract a noun phrase best suited as the concept title (1-8 words); if
  none can be extracted, return an empty string
- reason: explain the choice in one sentence; if unsure, still give a short note

Only output JSON in this format:
{{"text_type": "concept", "title_hint": "title hint or empty string", "reason": "one-sentence reason or null"}}"""


def _legacy_validation_task_prompt(
    *,
    title: str,
    source_text: str,
    qa_history: str = "",
) -> str:
    """Design one concrete validation task that checks real understanding rather
    than memorisation. Returns JSON validated by :class:`ValidationTask`.
    """
    history_text = qa_history or "(no usable Q&A history yet)"
    return f"""Basic learning is done. Design a "validation task" that checks whether the learner truly understood the concept instead of memorizing it.

Concept: {title}
Source text:
{source_text}
Previous Q&A history:
{history_text}

Requirements:
- The task must be impossible to fake by reciting the answer: e.g., "explain it in one
  sentence to someone who has never heard of it" (Feynman), "judge whether a given
  example contains this phenomenon and say why", "predict the outcome of a scenario", etc.
- Give exactly one task — don't split it into multiple-choice, don't list options
- target must spell out "the key points a correct understanding should include", to be
  used later to judge whether the answer is adequate
- Natural tone, like a friend handing them a small challenge

Only output JSON in this format:
{{"task": "validation task (specific and doable)", "target": "key points a correct understanding should include, 1-2 sentences"}}"""


def _legacy_validate_answer_prompt(
    *,
    title: str,
    task: str,
    target: str,
    answer: str,
    attempts_left: int | None = None,
) -> str:
    """Judge the learner's validation-task answer. Returns JSON validated by
    :class:`LegacyValidateAnswerResult`.
    """
    attempt_note = (
        f"\n- This is the last chance (can still retry {attempts_left} more times); "
        f"make the feedback warmer and more encouraging"
        if attempts_left is not None and attempts_left <= 1
        else ""
    )
    return f"""Judge whether the learner's answer to the "validation task" reflects genuine understanding.

Concept: {title}
Validation task: {task}
Key points a correct understanding should include: {target}
The learner's answer: {answer}

Judging rules:
- Does the answer cover the key points in target? If the direction is right and they can
  reason it through in their own words, is_correct is true
- No need for word-for-word matching; what matters is "can the learner say it in their own words"
- On a correct answer, feedback shows appreciation and extension, like a friend chatting
- On a wrong or off-topic answer, gently point out the gap; missing names in one sentence
  which key point was left out
- Overall tone: warm, non-judgmental, less exam-like{attempt_note}

Only output JSON in this format:
{{"is_correct": true or false, "feedback": "conversational feedback (1-2 sentences)", "missing": "the missing key point or null"}}"""


def simplify_explanation_prompt(
    *,
    title: str,
    source_text: str,
    explanation: str = "",
) -> str:
    """Rewrite the explanation (or the concept itself) one notch simpler, in the
    most everyday language. Returns plain text.
    """
    prev = f"(The learner still finds it too hard; the previous explanation was:\n{explanation})" if explanation else ""
    return f"""The learner says "that's still not simple enough." Bring the explanation down another notch and re-explain it in the most everyday plain language.

Concept: {title}
Source excerpt:
{source_text}
{prev}

Requirements:
- Use even simpler everyday language than last time, like explaining to a friend
- First state the core meaning in one sentence, then use a concrete daily example (bubble tea, food delivery, running a small shop's books)
- Lighten the tone; you may self-deprecatingly admit "I might have been a bit roundabout earlier" to lower the pressure
- End by gently asking "does that flow better now?" and stop — no more questions
- Plain text, no JSON"""


_DEEPER_QUESTION_GUIDES: dict[str, str] = {
    "verification_plus": (
        "Verify understanding once more with a new scenario or counterexample they haven't "
        "seen, to check whether they can recognize the concept in a shifted context; be concrete."
    ),
    "connection": (
        "Bring up a related concept they learned before and ask \"what connects or separates "
        "it from X you learned earlier?\" to help them weave new knowledge into a web."
    ),
    "counterfactual": (
        "Run a counterfactual: if this concept didn't exist, what would happen? Lead them "
        "from the opposite side to see the concept's value."
    ),
    "action": (
        "Push the concept into a concrete decision: \"if you had to make a related decision "
        "in real life tomorrow, how would you use it?\" Land the concept on real choices."
    ),
    "first_principles": (
        "Guide them to break the concept down to irreducible basic facts and re-derive it "
        "from the ground up, setting aside memorized conclusions."
    ),
}

_DEEPER_QUESTION_NAMES = {
    "verification_plus": "Re-verify",
    "connection": "Connection",
    "counterfactual": "Counterfactual",
    "action": "Action",
    "first_principles": "First principles",
}

DEEPER_QUESTION_ORDER: tuple[str, ...] = (
    "verification_plus",
    "connection",
    "counterfactual",
    "action",
    "first_principles",
)


def _legacy_deeper_question_prompt(
    *,
    title: str,
    source_text: str,
    question_type: str,
    qa_history: str = "",
) -> str:
    """Build one deeper-probing question of the given type. Returns plain text."""
    if question_type not in _DEEPER_QUESTION_GUIDES:
        raise ValueError(
            f"question_type must be one of {sorted(_DEEPER_QUESTION_GUIDES)}, "
            f"got {question_type}"
        )
    parts = [
        f"Current concept being learned: {title}",
        f"Source excerpt:\n{source_text}",
    ]
    if qa_history:
        parts.append(f"The learner's previous answers:\n{qa_history}")
    parts.append(f"【Deeper probe: {_DEEPER_QUESTION_NAMES[question_type]}】{_DEEPER_QUESTION_GUIDES[question_type]}")
    parts.append(
        "Requirement: output only this one deeper question — concrete, vivid, slightly "
        "challenging but not harsh, with no explanation."
    )
    return "\n\n".join(parts)


# ----------------------------------------------------- V0.3.0 Learning Loop v2
# 核心：持续观察 Learner State，用最小干预推动理解跃迁，不再靠固定轮次提问。
# 新 prompt 用 {name} 占位符（见 _fill）；JSON 示例中的花括号无需转义。

# V0.3.1 hotfix — 深化追问按回答丰富度分级：回答越简短，追问越生活化；
# 回答越深入，追问越往前推（类比 → 联系 → 反事实）。
DEEPENING_LEVEL_GUIDES: dict[str, str] = {
    "simple": "Everyday analogy: compare it to something familiar in daily life and ask \"what is this like?\" so the concept lands close to home.",
    "moderate": "Connection: link the concept to ones they've learned or to real-life scenes, asking \"what does this relate to?\"",
    "rich": "Counterfactual: have them imagine \"what would happen if this concept didn't exist?\" to see its value from the opposite side.",
}


def _fill(template: str, **kwargs: str) -> str:
    """Replace ``{name}`` placeholders. JSON braces in the template are safe."""
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", value)
    return template


def validation_task_prompt(
    *,
    source_text: str,
    concept: str,
    text_type: str = "",
    reading_answers: str = "",
) -> str:
    """Generate one low-cost task that verifies real understanding (not rote
    recall). Returns JSON validated by :class:`ValidationTask`."""
    reading_block = (
        f"\nThe learner's own understanding while reading (reference only; the task "
        f"should target what they haven't grasped):\n{reading_answers}"
        if reading_answers
        else ""
    )
    return _fill(
        """You are RecallOS's learning-task designer.

Based on source_text, create a validation task that determines whether the user truly
understands the content rather than just memorizing the source.

Prefer the task type that best fits the content, for example:
- summary: summarize in your own words
- translation: explain the jargon
- analogy: explain with an everyday example
- application: say how to apply it
- comparison: compare two concepts
- prediction: predict what happens in a given situation

Rules:
1. The task must be answerable on its own.
2. Prefer requiring the user to use their own words.
3. Avoid asking for a direct recitation of the source.
4. Difficulty: 1=basic understanding, 2=relational understanding, 3=higher-order understanding.
5. Generate only ONE task.
6. Only output valid JSON.

Input:
concept: {concept}
source_text: {source_text}
text_type: {text_type}{reading_block}

Output JSON format:
{"task": "In your own words — not the words from the source — explain why this concept matters.", "type": "summary", "difficulty": 2}""",
        source_text=source_text,
        concept=concept,
        text_type=text_type or "concept",
        reading_block=reading_block,
    )


def learner_state_analyzer_prompt(
    *,
    source_text: str,
    concept: str,
    task: str,
    user_answer: str,
    context: str = "",
    learning_goal: str = "understand",
    stuck_points: str = "",
    confidence_prediction: str = "",
) -> str:
    """Analyse the learner's closed-book explanation into a Learner State
    snapshot. Returns JSON validated by :class:`LearnerStateAnalysis`."""
    context_block = f"\n\nPrevious conversation:\n{context}" if context else ""
    goal_hint = {
        "understand": "Goal: understand the concept itself. Reaching relationship (can explain how the concept relates to others / its conditions) is enough.",
        "connect": "Goal: build connections. The learner must relate the concept to other concepts, scenarios, or conditions.",
        "apply": "Goal: apply it in practice. Reaching application (can say how to use it in a real situation) is enough.",
        "exam": "Goal: master it for an exam. Requires precise expression, accurate terms, and no omissions.",
    }.get(learning_goal, "Goal: understand the concept itself.")
    stuck_block = f"\nPlaces the learner said they got stuck while reading (use these to fill uncertain):\n{stuck_points}" if stuck_points else ""
    confidence_block = f"\nLearner's self-assessment before explaining: {confidence_prediction} (you may use it to gauge calibration)" if confidence_prediction else ""
    return _fill(
        """You are RecallOS's learner-state analyzer.

Task: based on the source text and the learner's closed-book explanation just given,
judge what they truly understand, where they're uncertain, and where they have misconceptions.
Don't just check keywords; judge whether they actually expressed the concept's meaning.

Understanding levels:
- surface: mostly parroting the source/keywords, no clear understanding of their own
- relationship: can explain how the concept relates to other concepts, conditions, or outcomes
- application: can explain how the concept applies to real situations
- essence: can explain why the concept holds, what problem it solves, or its underlying logic

{goal_hint}

Rules:
1. Base judgment on source_text; don't substitute the source's meaning with common sense.
2. Different wording with correct meaning counts as understanding.
3. Write core misconceptions into misconceptions.
4. Don't lower the level just because the answer is short.
5. Keep the learner's self-reported stuck points in uncertain when possible.
6. understood / uncertain / misconceptions should each list only the most important 1-3 items, in short phrases.
7. Only output valid JSON, no Markdown or explanation.

Input:
concept: {concept}
source_text: {source_text}
task: {task}
user_answer: {user_answer}
Learning goal: {goal_hint}{stuck_block}{confidence_block}{context_block}

Output JSON format:
{"understanding_level": "relationship", "understood": ["understands the relationship between choice and what's given up"], "uncertain": ["hasn't clearly understood that opportunity cost is only about the best alternative"], "misconceptions": [], "last_response_quality": "partial"}""",
        concept=concept,
        source_text=source_text,
        task=task,
        user_answer=user_answer,
        context_block=context_block,
        goal_hint=goal_hint,
        stuck_block=stuck_block,
        confidence_block=confidence_block,
    )


def validation_feedback_prompt(
    *,
    concept_title: str,
    user_answer: str,
    task_description: str,
    reading_answers: list[dict] | None = None,
) -> str:
    """生成验证反馈 —— 具体、可行动、有对比（纯文本，非 JSON）。

    用户在验证阶段答完题（且存在理解缺口）后立即展示，用来解释
    「为什么没到位」，而不是只说一句「理解不到位」。
    """
    reading_context = ""
    if reading_answers:
        answers_text = ", ".join(
            f"paragraph {i + 1}: {a['answer']}" for i, a in enumerate(reading_answers)
        )
        reading_context = f"The learner's understanding while reading: {answers_text}\n"
    return _fill(
        """You are a Socratic mentor. The learner is studying "{concept_title}".

Validation task:
{task_description}

{reading_context}The learner's answer:
{user_answer}

Evaluate the answer and give feedback that is **specific, actionable, and comparative**.

Evaluation dimensions:
1. **Core concept**: did they capture the essence of "{concept_title}"? (yes/partially/no)
2. **Expression**: did they explain it in their own words? (yes/partially/no)
3. **Perspective richness**: did they approach it from multiple angles? (yes/partially/no)

Feedback format (follow this structure strictly):
---
**Your answer mentioned:**
[list the valid points from the learner's answer, using • bullets]

**But the core of {concept_title} is:**
[state the essence of the concept in one sentence]

**Compare:**
You said "[the learner's wording]"
While a more accurate way to say {concept_title} is "[correct wording]"

**Try this example:**
[give a specific example related to the learner's answer]

**So:**
[summarize how well they understand: basically there / right direction but needs adjusting / needs to relearn]

---
**Important rules:**
- Never just say "right" or "wrong"
- Never just say "not quite there" without explaining why
- Always give a concrete comparison (what they said vs. what's correct)
- Always give a concrete example
- Sound like a friend, not an examiner
- Accept any correct wording of the learner's answer.""",
        concept_title=concept_title,
        user_answer=user_answer,
        task_description=task_description,
        reading_context=reading_context,
    )


def intervention_decider_prompt(
    *,
    source_text: str,
    concept: str,
    learner_state: str,
    current_target: str = "",
    mode: str = "validation",
    context: str = "",
    intervention_history: str = "",
    minimum_action: str = "",
    answer_richness: str = "",
) -> str:
    """Decide whether / how to intervene next (minimal intervention). Returns
    JSON validated by :class:`Intervention`.

    ``minimum_action`` (V0.3.0 patch 3) is the lowest allowed intensity: when
    the previous intervention failed to move the learner, the AI must pick
    something at least one ladder step above it.

    ``answer_richness`` (V0.3.1 hotfix) is the richness level of the learner's
    latest answer (simple/moderate/rich). The decider must then push with a
    question of that depth instead of a generic one.
    """
    context_block = f"\n\nPrevious conversation:\n{context}" if context else ""
    history_block = (
        f"\n\nThe learner's feedback on the interventions given (use it to pick the next type):\n{intervention_history}"
        if intervention_history
        else ""
    )
    richness_block = (
        f"\n\nRichness of the learner's last answer: {answer_richness} ({DEEPENING_LEVEL_GUIDES.get(answer_richness, '')})"
        if answer_richness in DEEPENING_LEVEL_GUIDES
        else ""
    )
    mode_hint = (
        "Current stage is validation: the goal is to confirm or repair understanding of the core meaning."
        if mode == "validation"
        else "Current stage is deepening: the goal is to find the one gap worth pushing on above the current understanding level."
    )
    if minimum_action in INTERVENTION_LABELS:
        extra = [
            (
                f"3. The last intervention did not help the learner progress: this time "
                f"the action must be at least as strong as \"{minimum_action}"
                f" ({INTERVENTION_LABELS[minimum_action]})\" — start from this level."
            )
        ]
        base = 4
    else:
        extra = []
        base = 3
    rules = [
        "1. Never use a heavier intervention when a lighter one will do (if a one-line hint works, don't give an example; if one example works, don't explain the full answer).",
        "2. Only escalate to a heavier intervention once the lighter one has failed.",
    ]
    rules += extra
    rules += [
        f"{base}. content should only give \"just enough to keep the learner thinking\" — never the full answer.",
        f"{base + 1}. When gaps remain but another intervention is no longer valuable, set action=none and finish.",
        f"{base + 2}. requires_user_response: true when the learner should answer again; false when you're directly giving a closing explanation.",
        f"{base + 3}. Use past intervention feedback: when the learner said a certain type (analogy/example/hint/counterexample) \"got much clearer\", prefer a simpler version of the same type; when they said \"still confused\", switch to a different type.",
        f"{base + 4}. Only output valid JSON.",
    ]
    rules_block = "\n".join(rules)
    return _fill(
        """You are RecallOS's minimal-intervention decider.

First answer: "What does this learner need most right now?" Then decide whether an
intervention is worthwhile and what the minimal one is.

Don't mechanically do "right → next question, wrong → ask again." Only intervene when
there is a real, valuable understanding gap.

Intervention intensity ladder (low to high; low = lighter):
hint (one-line hint) → example (concrete example) → analogy (analogy) → counterexample (counterexample) → question (direct question)

Mandatory rules:
{rules}

{mode_hint}

Input:
concept: {concept}
source_text: {source_text}
Current target: {current_target}
learner_state: {learner_state}{history_block}{context_block}{richness_block}

Output JSON format:
{"action": "counterexample", "reason": "the learner gets the basic relationship but can't identify the concept's boundaries", "content": "If you give up three options, is the opportunity cost all three added together? Why?", "requires_user_response": true}""",
        concept=concept,
        source_text=source_text,
        learner_state=learner_state,
        current_target=current_target or "validation task",
        mode_hint=mode_hint,
        rules=rules_block,
        history_block=history_block,
        context_block=context_block,
        richness_block=richness_block,
    )


def learner_state_updater_prompt(
    *,
    source_text: str,
    concept: str,
    intervention: str,
    user_answer: str,
    learner_state: str,
    context: str = "",
    learning_goal: str = "understand",
) -> str:
    """Update the Learner State from the learner's newest answer. Returns JSON
    validated by :class:`LearnerStateUpdate`."""
    context_block = f"\n\nPrevious conversation:\n{context}" if context else ""
    goal_hint = {
        "understand": "Goal: understand the concept itself; reaching relationship is enough.",
        "connect": "Goal: build connections; must be able to relate it to other concepts/scenarios.",
        "apply": "Goal: apply it in practice; reaching application is enough.",
        "exam": "Goal: master it for an exam; requires precise expression and accurate terms.",
    }.get(learning_goal, "")
    return _fill(
        """You are RecallOS's learner-state updater.

Update the learner state from the learner's latest answer to the last intervention.
Only give a fresh snapshot based on this newest answer; don't mechanically copy past
entries. The understanding level may rise, stay the same, or fall.

Understanding levels:
- surface: restating a definition or keywords
- relationship: explaining relationships between concepts
- application: being able to apply the concept to concrete situations
- essence: understanding the underlying logic, causal mechanism, or reason it exists

next_best_action is the AI's most valuable next move: none / hint / analogy / example / counterexample / question.

{goal_hint}

Rules:
1. Don't judge quality by answer length.
2. Correct but mechanical restating only counts as surface.
3. Only output valid JSON.

Input:
concept: {concept}
source_text: {source_text}
Last intervention: {intervention}
user_answer: {user_answer}
learner_state: {learner_state}{context_block}

Output JSON format:
{"understanding_level": "application", "understood": ["understands that opportunity cost relates to choices", "can spot opportunity cost in specific situations"], "uncertain": [], "misconceptions": [], "last_response_quality": "deep", "next_best_action": "none"}""",
        concept=concept,
        source_text=source_text,
        intervention=intervention,
        user_answer=user_answer,
        learner_state=learner_state,
        goal_hint=goal_hint,
        context_block=context_block,
    )


def deepening_offer_prompt(
    *,
    concept: str,
    understanding_level: str,
) -> str:
    """Ask the learner whether to keep going deeper after validation passes.
    Returns JSON validated by :class:`DeepeningOffer`."""
    return _fill(
        """You are RecallOS's learning coach.

The learner already basically understands "{concept}". Generate one natural, inviting
offer to go deeper, and let them choose.

Requirements:
1. It must convey "you already get the core."
2. Hint at the value of going deeper.
3. Don't use "do you want to".
4. No pressure.
5. At most 25 words.
6. Adjust the tone to understanding_level.
7. Only output valid JSON.

Input:
concept: {concept}
understanding_level: {understanding_level}

Output JSON format:
{"offer": "You've got the core — want to go one layer deeper and see why it holds?", "options": ["Go deeper", "That's enough"]}""",
        concept=concept,
        understanding_level=understanding_level,
    )


def deepening_question_prompt(
    *,
    level: str,
    concept: str,
    source_text: str,
    qa_history: str = "",
    understanding_level: str = "",
) -> str:
    """按回答丰富度分级生成一道深化追问（simple/moderate/rich）。纯文本，非 JSON。

    - ``simple``：生活类比「这就像什么？」
    - ``moderate``：联系类「这和什么有关？」
    - ``rich``：反事实「如果没有它会怎样？」
    """
    if level not in DEEPENING_LEVEL_GUIDES:
        raise ValueError(
            f"level must be one of {sorted(DEEPENING_LEVEL_GUIDES)}, got {level!r}"
        )
    history_block = f"\nThe learner's previous answers:\n{qa_history}" if qa_history else ""
    level_block = (
        f"\nCurrent understanding level: {understanding_level}" if understanding_level else ""
    )
    return _fill(
        """You are RecallOS's learning coach.

The learner just made an output about "{concept}". Based on the richness of that answer,
craft a deeper follow-up question that moves them exactly one step forward.

{guide}

Requirements:
1. Output only this one deeper question — concrete, vivid, slightly challenging but not harsh.
2. Follow the level guide above strictly; don't skip levels.
3. No explanations, headings, or JSON.

Input:
concept: {concept}
source_text: {source_text}{level_block}{history_block}""",
        concept=concept,
        source_text=source_text,
        guide=DEEPENING_LEVEL_GUIDES[level],
        level_block=level_block,
        history_block=history_block,
    )


# ------------------------------------------------------------------- messages

def build_messages(user_content: str) -> list[dict[str, str]]:
    """Assemble the system + user messages for the DeepSeek client."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ----------------------------------------------------------------- JSON helpers

def parse_json_response(text: str) -> Any:
    """Strip markdown fences / surrounding noise and return the parsed JSON."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    candidates = [
        i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1
    ]
    start = min(candidates) if candidates else -1
    if start == -1:
        raise ValueError("no JSON object or array found in response")
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        raise ValueError("unbalanced JSON delimiters in response")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in response: {exc}") from exc


def validate_response(text: str, model: type[T]) -> T:
    """Parse AI JSON output and validate it against a Pydantic model."""
    return model.model_validate(parse_json_response(text))


def validate_response_list(text: str, model: type[T]) -> list[T]:
    """Parse an AI JSON array and validate each item against a Pydantic model."""
    data = parse_json_response(text)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array")
    return [model.model_validate(item) for item in data]


# ------------------------------------------------------------ response models

class CheckAnswerResult(RecallBaseModel):
    """Validated output of :func:`check_answer_prompt`."""

    is_correct: bool
    feedback: NonEmptyStr
    hint: OptionalStr = None


class SummaryResult(RecallBaseModel):
    """Validated output of :func:`summary_prompt`."""

    breakthrough: NonEmptyStr
    plain: OptionalStr = None
    tomorrow_hook: NonEmptyStr


class ConnectionSuggestion(RecallBaseModel):
    """One item of the validated output of :func:`connections_prompt`."""

    concept_title: NonEmptyStr
    relation_text: NonEmptyStr


class TextTypeResult(RecallBaseModel):
    """Validated output of :func:`detect_text_type_prompt`."""

    text_type: Literal[
        "concept", "definition", "article", "list", "question", "other"
    ]
    title_hint: OptionalStr = None
    reason: OptionalStr = None


class LegacyValidationTask(RecallBaseModel):
    """Validated output of the legacy (V0.3.0 pre-refactor) validation prompt."""

    task: NonEmptyStr
    target: NonEmptyStr


class LegacyValidateAnswerResult(RecallBaseModel):
    """Validated output of the legacy validate-answer prompt."""

    is_correct: bool
    feedback: NonEmptyStr
    missing: OptionalStr = None


class ValidationTask(RecallBaseModel):
    """Validated output of the V0.3.0 Learning Loop v2 validation prompt."""

    task: NonEmptyStr
    type: NonEmptyStr
    difficulty: int = Field(default=2, ge=1, le=3)


UnderstandingLevel = Literal["surface", "relationship", "application", "essence"]
ResponseQuality = Literal["deep", "partial", "shallow"]
InterventionAction = Literal[
    "none", "hint", "analogy", "example", "counterexample", "question", "rephrase"
]

# V0.3.0 patch 3 — intervention intensity ladder, low -> high (强制优先级).
# Only escalate to a heavier intervention once the lighter one has failed.
INTERVENTION_LADDER: tuple[str, ...] = (
    "hint",
    "example",
    "analogy",
    "counterexample",
    "question",
)
INTERVENTION_LABELS: dict[str, str] = {
    "hint": "One-line hint",
    "example": "Concrete example",
    "analogy": "Analogy",
    "counterexample": "Counterexample",
    "question": "Direct question",
}


class LearnerStateAnalysis(RecallBaseModel):
    """Validated output of :func:`learner_state_analyzer_prompt`."""

    understanding_level: UnderstandingLevel
    understood: list[str] = Field(default_factory=list)
    uncertain: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    last_response_quality: ResponseQuality


class Intervention(RecallBaseModel):
    """Validated output of :func:`intervention_decider_prompt`."""

    action: InterventionAction
    reason: OptionalStr = None
    content: NonEmptyStr
    requires_user_response: bool = True


class LearnerStateUpdate(LearnerStateAnalysis):
    """Validated output of :func:`learner_state_updater_prompt`."""

    next_best_action: InterventionAction


class DeepeningOffer(RecallBaseModel):
    """Validated output of :func:`deepening_offer_prompt`."""

    offer: NonEmptyStr
    options: list[NonEmptyStr] = Field(default_factory=lambda: ["Go deeper", "That's enough"])


__all__ = [
    "SYSTEM_PROMPT",
    "question_prompt",
    "opening_question_prompt",
    "check_answer_prompt",
    "summary_prompt",
    "connections_prompt",
    "reference_answer_prompt",
    "simplify_question_prompt",
    "angle_shift_prompt",
    "explain_prompt",
    "warmup_prompt",
    "build_messages",
    "parse_json_response",
    "validate_response",
    "validate_response_list",
    "CheckAnswerResult",
    "SummaryResult",
    "ConnectionSuggestion",
    "TextTypeResult",
    "ValidationTask",
    "LearnerStateAnalysis",
    "Intervention",
    "LearnerStateUpdate",
    "DeepeningOffer",
    "detect_text_type_prompt",
    "validation_task_prompt",
    "learner_state_analyzer_prompt",
    "validation_feedback_prompt",
    "intervention_decider_prompt",
    "learner_state_updater_prompt",
    "deepening_offer_prompt",
    "deepening_question_prompt",
    "DEEPENING_LEVEL_GUIDES",
    "simplify_explanation_prompt",
    "DEEPER_QUESTION_ORDER",
    "LegacyValidationTask",
    "LegacyValidateAnswerResult",
    "_legacy_validation_task_prompt",
    "_legacy_validate_answer_prompt",
    "_legacy_deeper_question_prompt",
    "UnderstandingLevel",
    "ResponseQuality",
    "InterventionAction",
    "INTERVENTION_LADDER",
    "INTERVENTION_LABELS",
]
