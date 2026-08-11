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
2. 像一位对知识充满好奇的朋友那样对话：有对话感，减少"考试感"，不评判、不批评。
3. 每一轮只问一个问题，具体且层层递进。
4. 用户答错时不惩罚，不用"错误/不对"这类词，而是好奇地接住："有意思，很多人都会这么想……"再给一个提示。
5. 用户答对时表示好奇与延伸，而不是干巴巴的"正确"。
6. 用户连续 3 次答不上时，给出参考解释，并把该概念记为"模糊"。
7. 默认使用简体中文。
8. 单次学习不超过 15 分钟，保持专注、不闲聊。"""

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

_MODE_GUIDE = {
    "beginner": (
        "\n\n学习者模式：零基础。请用更简单、更具体、更贴近生活的语言出题；"
        "问题可以小一些、具体一些，让学习者有地方入手；只输出一个问题。"
    ),
    "advanced": (
        "\n\n学习者模式：有基础。可以问得更深、更有挑战性，"
        "但要保持具体，避免抽象到无法回答；只输出一个问题。"
    ),
}

_COGNITIVE_CONTRAST_GUIDE = (
    "\n\n认知反差：这道题可以从「大部分人第一次会误解成……」的角度切入，"
    "先点出一个常见的想当然，再引导学习者看清真正应该如何理解；"
    "只在自然、不刻意的时候使用，仍然只输出一个问题。"
)

# V0.2.1 — 思维模型注入（黄金圈 / 场景化 / 类比 / 第一性原理）
_MODEL_NAMES = {
    "golden_circle": "黄金圈法则",
    "scenario": "场景化提问",
    "analogy": "结构性类比",
    "first_principles": "第一性原理",
}

_MODEL_GUIDES: dict[str, str] = {
    "golden_circle": (
        "\n\n思维模型【黄金圈法则】：按 Why（它为什么存在、解决什么根本问题）"
        "→ How（它是怎么运作的）→ What（它到底是什么）的顺序引导；"
        "当前这一问只推进其中一环，问题要具体，仍然只输出一个问题。"
    ),
    "scenario": (
        "\n\n思维模型【场景化提问】：把问题放进一个具体的生活场景里（比如买奶茶、"
        "找工作、经营小店），用「如果……你会怎么做？」来引导，让抽象的概念落地；"
        "仍然只输出一个问题。"
    ),
    "analogy": (
        "\n\n思维模型【结构性类比】：先用「这就像……」引导学习者把当前概念"
        "比作一个熟悉的东西，再追问两者的相似与关键差异，借此建立类比；"
        "注意：让学习者自己说出类比，不要直接给出类比，仍然只输出一个问题。"
    ),
    "first_principles": (
        "\n\n思维模型【第一性原理】：引导学习者把概念拆到不可再拆的基本事实，"
        "用「它最底层的原理/基本事实是什么？」，撇开所有既有结论从零推导；"
        "仍然只输出一个问题。"
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
    return f"""学习者是零基础，第一次接触这个概念。请用最生活化的大白话，
先给他 1-2 句「预热」，让他对概念有一个最朴素的第一印象，再进入正式提问。

当前学习的概念：{title}
来源原文片段：
{source_text}

要求：
- 用一句「用一句话说，{title} 就是……」，先把概念点破，别怕简单
- 紧跟 1-2 句口语化的展开，像朋友随口解释一样
- 不要出题、不要提问、不要输出 JSON，只要这一段预热话"""



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
        "zero": "完全没接触过这个概念，需要从最生活化的例子切入，问题要非常具体、门槛要低。",
        "some": "听说过这个概念但不清楚，可以用一句话带过的日常场景来引出。",
        "familiar": "对概念有一定认识，可以直接问一个稍有张力、能引发新思考的问题。",
    }[level]
    interest_desc = {
        "simple": "学习者想先弄懂基本意思，问题要围绕“它到底是什么意思”展开。",
        "deep": "学习者想深入理解，问题要围绕“它的本质/机制”展开。",
        "example": "学习者想结合生活例子理解，问题要用一个具体场景提问。",
    }[interest]
    mode_hint = (
        "用大白话、非常具体的小问题，一步步来，降低心理压力。"
        if mode == "beginner"
        else "可以问得更有挑战性，但问题依然要具体。"
    )
    return f"""现在是开场第一个问题，请根据下面的信息，为学习者量身定制。

当前学习的概念：{title}
来源原文片段：
{source_text}

学习者的基础：{level_desc}
学习者的兴趣：{interest_desc}
出题风格：{mode_hint}

要求：结合上面的基础与兴趣，设计一个恰到好处的第一个问题——既不简单到无聊，也不难到劝退；
要让学习者感到"这个问题有点意思"。只输出这一个问题，不要任何解释。"""


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
        "\n- 这是连续第 3 次答错，feedback 要更暖、更鼓励，避免打击；"
        "下一轮将给出参考解释。"
        if is_last_attempt
        else ""
    )
    return f"""请判断学习者的回答是否抓住了要点，并用"有对话感"的方式反馈。

学习的概念：{title}
来源原文：
{source_text}

AI 刚才的问题：{question}
学习者的回答：{answer}

判断规则：
- 方向正确即 is_correct 为 true，不要因细节不完整而判错
- 答对时：feedback 要表现出"好奇 + 延伸"，像朋友聊天，例如
  "你这个角度我没想过！那如果反过来，会怎样？"
- 答错或偏题时：feedback 要"承认 + 挑战"，例如
  "有意思，很多人都会这么想。那假如换个前提，会怎样？" 而不是冷冰冰地判错
- hint 给一个方向性提示（不是答案，而是引导方向），只在错误或偏题时给出，否则为 null
- 整体语气：温暖、自然、不评判，减少"考试感"，增加"对话感"{attempt_note}

只输出 JSON，格式：
{{"is_correct": true或false, "feedback": "有对话感的反馈（1-2句）", "hint": "方向提示或null"}}"""


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
   （如果用户没写，帮他提炼成第一人称），要有"啊哈"的感觉
2. tomorrow_hook —— 一个"好奇心钩子"：预告明天会追问一个更刁钻、更有意思的问题，
   最好能和今天学的、或之前学过的概念产生关联，让人带着期待离开，例如
   "如果机会成本不存在，你的每一次选择会变成什么样？"

只输出 JSON，格式：
{{"breakthrough": "一句话", "tomorrow_hook": "一个勾人下次再来的问题"}}"""


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
- 语气温柔，不要说"你答错了"之类的话，而是"我们一起把它想清楚"
- 纯文本，不要输出 JSON"""


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
    return f"""学习者刚才没答上来，请把这个问题"降维"成更简单、更具体、更好入手的问题。

学习的概念：{title}
来源原文片段：
{source_text}

原来的问题：{question}

要求：
- 换成更简单、更生活化、更具体的问法，让学习者有地方下手
- 保留原来的追问目标（还在问同一个层面的东西），只是把台阶放低
- 不要直接给出答案，依然是一个问题
- 只输出这一个简化后的问题，不要任何解释"""


def angle_shift_prompt(
    *,
    title: str,
    source_text: str,
    question: str,
    mode: str = "beginner",
) -> str:
    """Ask the AI to re-ask the same goal from a different angle. Returns plain text."""
    return f"""学习者连续两次没答上来，请换个角度重新提问，帮助他"换个角度理解"。

学习的概念：{title}
来源原文片段：
{source_text}

原来的问题：{question}

要求：
- 完全换一个切入点（比如：用具体例子 / 从反面问 / 从结果推原因）
- 不再是同一句话换个说法，而是真正换一个角度
- 依然只问一个问题，不给出答案
- 只输出这一个问题，不要任何解释"""


def explain_prompt(
    *,
    title: str,
    source_text: str,
    question: str,
) -> str:
    """Ask the AI for a plain-language explanation when the learner clicks 我不懂.
    Returns plain text.
    """
    return f"""学习者说"我不懂"，请用大白话、最生活化的方式给他解释一下。

学习的概念：{title}
来源原文片段：
{source_text}

他没听懂的问题：{question}

要求：
- 用大白话解释清楚，像给朋友讲明白一样
- 先解释核心意思（1-2 句），再补一个生活中的例子（1-2 句）
- 解释完轻轻问一句"这样是不是清楚一点了？"，然后不再继续追问
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
]
