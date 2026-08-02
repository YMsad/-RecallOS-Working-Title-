"""Prompt templates for the Socratic learning flow.

Follows PRD V0.1's four-layer questioning structure (核心 / 重要 / 反事实 /
连接). ``question_prompt`` returns plain text; the others return structured
JSON, validated by Pydantic models in this module.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from core.models import NonEmptyStr, OptionalStr, RecallBaseModel

T = TypeVar("T", bound=RecallBaseModel)

# ----------------------------------------------------------------- role prompt

SYSTEM_PROMPT = """你是一位永不疲倦的苏格拉底式学习伙伴，名叫 RecallOS。
你的目标不是替用户"记住"，而是帮用户"想通"。

核心原则：
1. 绝不直接给出答案，只给方向性提示，引导用户自己说出结论。
2. 语气温和、鼓励、不评判。
3. 每一轮只问一个问题，具体且层层递进。
4. 用户答错时不惩罚，只说"再想想"并给一个提示。
5. 用户连续 3 次答不上时，给出参考解释，并把该概念记为"模糊"。
6. 默认使用简体中文。
7. 单次学习不超过 15 分钟，保持专注、不闲聊。"""

# ------------------------------------------------------------------ four layers

_LAYER_GUIDES: dict[int, str] = {
    1: (
        "第一层追问——引导学习者用自己的话概括这个概念的核心是什么。\n"
        "要求：问题要具体，避免\"什么是X\"这类可以直接照书背诵的提问；"
        "只输出这一个问题，不要任何解释。"
    ),
    2: (
        "第二层追问——引导学习者说清楚这个概念为什么重要：它解决了什么难题，"
        "不理解它会带来什么后果。\n要求：问题要落到具体场景，只输出这一个问题。"
    ),
    3: (
        "第三层追问——反事实推演：假如没有这个概念（它不存在），会发生什么？\n"
        "要求：从反面出发，引导学习者看清概念的价值，只输出这一个问题。"
    ),
    4: (
        "第四层追问——结合 {related_concepts}，问一个"
        "\"它和你之前学的 X 有什么联系\"的问题。\n"
        "要求：只挑最有价值的一个连接点，问题要具体，只输出这一个问题。"
    ),
}

_LAYER_NAMES = {1: "核心", 2: "重要", 3: "反事实", 4: "连接"}


def question_prompt(
    layer: int,
    *,
    title: str,
    source_text: str,
    qa_history: str = "",
    related_concepts: list[str] | None = None,
) -> str:
    """Build the user prompt for one layer of questioning. Returns plain text."""
    if layer not in _LAYER_GUIDES:
        raise ValueError(f"layer must be one of {sorted(_LAYER_GUIDES)}, got {layer}")

    if layer == 4:
        related = "、".join(related_concepts) if related_concepts else "你之前学过的其他概念"
        guide = _LAYER_GUIDES[4].format(related_concepts=related)
    else:
        guide = _LAYER_GUIDES[layer]

    parts = [
        f"当前学习的概念：{title}",
        f"来源原文片段：\n{source_text}",
    ]
    if qa_history:
        parts.append(f"学习者此前的回答历史：\n{qa_history}")
    parts.append(f"【{_LAYER_NAMES[layer]}层追问】{guide}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- check answer

def check_answer_prompt(
    *,
    title: str,
    source_text: str,
    question: str,
    answer: str,
) -> str:
    """Ask the AI to judge whether the learner's answer hit the point. Returns JSON."""
    return f"""请判断学习者的回答是否抓住了要点。

学习的概念：{title}
来源原文：
{source_text}

AI 刚才的问题：{question}
学习者的回答：{answer}

判断规则：
- 方向正确即 is_correct 为 true，不要因细节不完整而判错
- 正确时 feedback 用 1 句话肯定（不要重复答案）
- 错误或偏题时，hint 给一个方向性提示（不是答案，而是引导方向）
- hint 只有在错误或偏题时才给出，否则为 null

只输出 JSON，格式：
{{"is_correct": true或false, "feedback": "简短肯定或纠正（1-2句）", "hint": "方向提示或null"}}"""


# ------------------------------------------------------------ daily summary

def summary_prompt(
    *,
    title: str,
    user_definition: str,
    qa_history: str,
) -> str:
    """Ask the AI to generate the daily summary. Returns JSON."""
    return f"""今天的学习结束了。请根据以下内容生成每日总结。

学习的概念：{title}
用户的最终理解：{user_definition}
追问记录：
{qa_history}

生成两项内容：
1. breakthrough —— 一句"我终于搞懂了……"风格的话，用学习者自己的话总结今天最大的收获
   （如果用户没写，帮他提炼成第一人称）
2. tomorrow_hook —— 一个明天继续追问的问题，尽量与之前学过的概念挂钩，制造期待感

只输出 JSON，格式：
{{"breakthrough": "一句话", "tomorrow_hook": "一个问题"}}"""


# ------------------------------------------------------------------ connections

def connections_prompt(
    *,
    title: str,
    source_text: str,
    all_concepts: list[str],
) -> str:
    """Ask the AI to recommend knowledge connections. Returns JSON array."""
    concepts_text = "\n".join(f"- {c}" for c in all_concepts) or "（暂无）"
    return f"""请为当前概念推荐可能的知识连接。

当前概念：{title}
概念要点：{source_text}
用户目前已学过的概念列表：
{concepts_text}

找出 2-3 个与当前概念最有关系（同类 / 对比 / 因果 / 相关）的概念，
并为每一条写清楚两者是什么关系。
concept_title 必须是上面列表中出现过的标题。

只输出 JSON 数组，格式：
[{{"concept_title": "已学概念的标题", "relation_text": "两者的关系说明"}}]"""


# ------------------------------------------------------------------ reference

def reference_answer_prompt(
    *,
    title: str,
    source_text: str,
    question: str,
    attempts: int,
) -> str:
    """Ask the AI to give a clear reference explanation. Returns plain text."""
    return f"""学习者在连续 {attempts} 次没能答上以下问题，请给出一段清晰、完整的参考解释。

学习的概念：{title}
来源原文：
{source_text}

AI 刚才的问题：{question}

要求：
- 直接给出概念的正确解释并回答这个问题，帮学习者理解
- 简洁（3-5 句），直接陈述，不要再使用提问方式
- 纯文本，不要输出 JSON"""


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
    tomorrow_hook: NonEmptyStr


class ConnectionSuggestion(RecallBaseModel):
    """One item of the validated output of :func:`connections_prompt`."""

    concept_title: NonEmptyStr
    relation_text: NonEmptyStr


__all__ = [
    "SYSTEM_PROMPT",
    "question_prompt",
    "check_answer_prompt",
    "summary_prompt",
    "connections_prompt",
    "reference_answer_prompt",
    "build_messages",
    "parse_json_response",
    "validate_response",
    "validate_response_list",
    "CheckAnswerResult",
    "SummaryResult",
    "ConnectionSuggestion",
]
