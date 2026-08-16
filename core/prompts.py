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
        "问题可以小一些、具体一些，让学习者有地方入手；"
        "问题要有画面感、有点意思，避免枯燥的教科书式提问；只输出一个问题。"
    ),
    "advanced": (
        "\n\n学习者模式：有基础。可以问得更深、更有挑战性，"
        "但要保持具体，避免抽象到无法回答；"
        "问题要有张力、有画面感，让学习者觉得'这个角度我没想过'；只输出一个问题。"
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

def review_question_prompt(
    *,
    title: str,
    source_text: str,
) -> str:
    """Ask the AI to craft a tough, provocative review question for a concept."""
    return f"""今天是复习日。请为这个概念出一道刁钻、有挑战性但不刁难人的复习题。

学习的概念：{title}
来源原文：
{source_text}

要求：
- 问题要能检验「学习者是真的懂了，还是只是背下来了」
- 可以结合反事实（假如…会怎样）、认知反差、或和生活场景结合
- 只输出这一个问题，不要任何解释、不要 JSON"""


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


# --------------------------------------------------------------- V0.3.0 flow
# 底层逻辑重构新增的 prompt：文本类型识别、验证任务、验证判定、降维解释、
# 深化追问。旧 prompt 全部保留；返回 JSON 的结构均由本模块的模型校验。

_TEXT_TYPE_CHOICES = {
    "concept": "一个独立的概念、术语（如「机会成本」）",
    "definition": "一段对某个概念或事物的解释性说明",
    "article": "一篇较长的文章、章节或资料",
    "list": "一组并列的要点、步骤或清单",
    "question": "一个问题、FAQ 或待解答的疑问",
    "other": "其他类型的学习材料，尽量少用",
}


def detect_text_type_prompt(*, raw_text: str) -> str:
    """Classify the pasted material before learning starts. Returns JSON
    validated by :class:`TextTypeResult`.
    """
    choices_text = "\n".join(f"- {k}：{v}" for k, v in _TEXT_TYPE_CHOICES.items())
    return f"""请先判断学习者粘贴的这段材料属于哪种类型，以便用合适的方式学习。

粘贴的材料原文：
{raw_text}

类型可选值：
{choices_text}

要求：
- 类型判断要贴近实际：独立术语/知识点选 concept；解释性段落选 definition；
  较长或结构不强的选 article；并列要点选 list（这种通常可拆成多个概念）；
  带问句的选 question；其余才选 other
- title_hint：从材料中提取一个最适合当作概念标题的名词短语（1-8 个字），
  提取不到就填空字符串
- reason：用一句话说明为什么这么判断，不确定时也给一句简短说明

只输出 JSON，格式：
{{"text_type": "concept", "title_hint": "标题提示或空字符串", "reason": "一句话理由或null"}}"""


def _legacy_validation_task_prompt(
    *,
    title: str,
    source_text: str,
    qa_history: str = "",
) -> str:
    """Design one concrete validation task that checks real understanding rather
    than memorisation. Returns JSON validated by :class:`ValidationTask`.
    """
    history_text = qa_history or "（暂无可用的追问记录）"
    return f"""基本学习已经完成。请设计一个「验证任务」，检验学习者是真的搞懂了概念，而不是背下来了。

学习的概念：{title}
来源原文：
{source_text}
此前的追问记录：
{history_text}

要求：
- 任务必须无法靠背答案蒙混：可以是「用一句话向完全没听过的人解释」（费曼）、
  「判断某个例子里是否发生了这个现象并说明理由」、「预测某个场景的结果」等
- 只给一个任务，不要拆成选择题，不要列出选项
- target 写清楚「一个正确理解应包含的关键点」，供后续判断学习者的回答是否到位
- 语气自然，像朋友递给他的一个小挑战

只输出 JSON，格式：
{{"task": "验证任务（具体、可执行）", "target": "正确理解应包含的关键点，1-2句"}}"""


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
        f"\n- 这是最后一次机会（还可重试 {attempts_left} 次），feedback 要更暖、更鼓励"
        if attempts_left is not None and attempts_left <= 1
        else ""
    )
    return f"""请判断学习者在「验证任务」中的回答是否体现了真正的理解。

学习的概念：{title}
验证任务：{task}
正确理解应包含的关键点：{target}
学习者的回答：{answer}

判断规则：
- 回答是否覆盖 target 里的关键点；方向对、能用学习者自己的话说得通道理即 is_correct 为 true
- 不必逐字逐句，重点是「学习者自己的话说得出」
- 答对时 feedback 表示欣赏与延伸，像朋友聊天
- 答错或偏题时 feedback 温和指出差距，missing 用 1 句话点出少了哪个关键点
- 整体语气温暖、不评判，减少考试感{attempt_note}

只输出 JSON，格式：
{{"is_correct": true或false, "feedback": "有对话感的反馈（1-2句）", "missing": "缺失的关键点或null"}}"""


def simplify_explanation_prompt(
    *,
    title: str,
    source_text: str,
    explanation: str = "",
) -> str:
    """Rewrite the explanation (or the concept itself) one notch simpler, in the
    most everyday language. Returns plain text.
    """
    prev = f"（学习者仍觉得不够简单，之前的解释是：\n{explanation}）" if explanation else ""
    return f"""学习者表示「讲得不够简单」，请把概念解释再降一个台阶，用最生活化的大白话重新讲一遍。

学习的概念：{title}
来源原文片段：
{source_text}
{prev}

要求：
- 用比上次更简单的生活化语言，像给朋友解释一样
- 先一句话说清核心意思，再用一个具体的日常例子（比如买奶茶、点外卖、开店算账）
- 语气轻一点，可以自嘲式地承认「我之前可能说得有点绕」，降低压力
- 结尾轻轻问一句「这样是不是顺多了？」，然后不要再提问
- 纯文本，不要输出 JSON"""


_DEEPER_QUESTION_GUIDES: dict[str, str] = {
    "verification_plus": (
        "用一个新的、他没见过的场景或反例再验证一次理解，"
        "看他能不能在变通的场合认出这个概念，问题要具体。"
    ),
    "connection": (
        "把他以前学过的相关概念拿出来，问一个「它和你之前学的 X 有什么联系或区别」"
        "的问题，引导他把新知识连成网。"
    ),
    "counterfactual": (
        "做反事实推演：假如没有这个概念（它不存在），会发生什么？"
        "从反面引导他看清这个概念的价值。"
    ),
    "action": (
        "把概念推进到一个具体的行动决策：「如果明天他要在真实生活里做一个相关的"
        "决定，他会怎么用上它？」让概念落到真实的选择上。"
    ),
    "first_principles": (
        "引导他把概念拆到不可再拆的基本事实，从最底层重新推导一遍，"
        "撇开背下来的结论。"
    ),
}

_DEEPER_QUESTION_NAMES = {
    "verification_plus": "再验证",
    "connection": "联系",
    "counterfactual": "反事实",
    "action": "行动",
    "first_principles": "第一性原理",
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
        f"当前学习的概念：{title}",
        f"来源原文片段：\n{source_text}",
    ]
    if qa_history:
        parts.append(f"学习者此前的回答记录：\n{qa_history}")
    parts.append(f"【{_DEEPER_QUESTION_NAMES[question_type]}层深化】{_DEEPER_QUESTION_GUIDES[question_type]}")
    parts.append(
        "要求：只输出这一个深化问题，具体、有画面感、略有挑战但不刁难，不要任何解释。"
    )
    return "\n\n".join(parts)


# ----------------------------------------------------- V0.3.0 Learning Loop v2
# 核心：持续观察 Learner State，用最小干预推动理解跃迁，不再靠固定轮次提问。
# 新 prompt 用 {name} 占位符（见 _fill）；JSON 示例中的花括号无需转义。

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
) -> str:
    """Generate one low-cost task that verifies real understanding (not rote
    recall). Returns JSON validated by :class:`ValidationTask`."""
    return _fill(
        """你是 RecallOS 的学习任务设计器。

根据 source_text 生成一个验证任务，用来判断用户是否真正理解内容，而不是单纯记忆原文。

优先选择最适合当前内容的任务类型，例如：
- summary：用自己的话概括
- translation：解释专业表达
- analogy：用生活例子解释
- application：说明如何应用
- comparison：比较两个概念
- prediction：预测某种情况下会发生什么

规则：
1. 任务必须能独立回答。
2. 优先要求用户使用自己的话。
3. 避免直接要求复述原文。
4. 难度：1=基础理解，2=关系理解，3=较高阶理解。
5. 只生成一个任务。
6. 只输出合法 JSON。

输入：
concept: {concept}
source_text: {source_text}
text_type: {text_type}

输出 JSON 格式：
{"task": "不用原文中的句子，用自己的话解释这个概念为什么重要。", "type": "summary", "difficulty": 2}""",
        source_text=source_text,
        concept=concept,
        text_type=text_type or "concept",
    )


def learner_state_analyzer_prompt(
    *,
    source_text: str,
    concept: str,
    task: str,
    user_answer: str,
    context: str = "",
) -> str:
    """Analyse the learner's closed-book explanation into a Learner State
    snapshot. Returns JSON validated by :class:`LearnerStateAnalysis`."""
    context_block = f"\n\n此前的对话记录：\n{context}" if context else ""
    return _fill(
        """你是 RecallOS 的学习者状态分析器。

任务：根据原文和用户刚给出的闭卷解释，判断用户真正理解了什么、哪里不确定、哪里存在误解。
不要只检查关键词；要判断用户是否真正表达了概念含义。

理解层级：
- surface：主要复述原文/关键词，没有明显自己的理解
- relationship：能解释概念与其他概念、条件或结果之间的关系
- application：能说明这个概念如何用于实际情况
- essence：能解释概念为什么成立、解决什么问题或底层逻辑

规则：
1. 以 source_text 为主要依据，不凭常识替换原文含义。
2. 表达方式不同但含义正确，应判为理解。
3. 有核心误解时写入 misconceptions。
4. 不因回答简短而降低层级判断。
5. understood / uncertain / misconceptions 各自只列最重要的 1-3 条，用短句。
6. 只输出合法 JSON，不要 Markdown 或解释。

输入：
concept: {concept}
source_text: {source_text}
task: {task}
user_answer: {user_answer}{context_block}

输出 JSON 格式：
{"understanding_level": "relationship", "understood": ["理解了选择与放弃之间的关系"], "uncertain": ["没有明确理解机会成本只关注最佳替代方案"], "misconceptions": [], "last_response_quality": "partial"}""",
        concept=concept,
        source_text=source_text,
        task=task,
        user_answer=user_answer,
        context_block=context_block,
    )


def intervention_decider_prompt(
    *,
    source_text: str,
    concept: str,
    learner_state: str,
    current_target: str = "",
    mode: str = "validation",
    context: str = "",
) -> str:
    """Decide whether / how to intervene next (minimal intervention). Returns
    JSON validated by :class:`Intervention`."""
    context_block = f"\n\n此前的对话记录：\n{context}" if context else ""
    mode_hint = (
        "当前处于验证阶段：目标是确认/修复对核心含义的理解。"
        if mode == "validation"
        else "当前处于深入阶段：目标是找到当前理解层级之上、最值得推进的那一个缺口。"
    )
    return _fill(
        """你是 RecallOS 的最小干预决策器。

先回答：「这个用户现在最需要什么？」再决定是否值得干预、用什么最小干预。

不要机械地"答对→下一题，答错→再问"。只有在存在真实且有价值的理解缺口时才干预。

最小干预优先级（越靠前越好）：
none → hint → example / analogy → counterexample → question → 解释

规则：
1. 能用一句提示解决，就不要给例子；能用一个例子解决，就不要解释完整答案。
2. content 只给"刚好让用户能继续思考"的内容，绝不直接给出完整答案。
3. 用户仍有缺口但干预已没有价值时，action=none 并结束。
4. requires_user_response：需要用户重新回答时为 true；直接给出收尾解释时为 false。
5. 只输出合法 JSON。

{mode_hint}

输入：
concept: {concept}
source_text: {source_text}
当前目标：{current_target}
learner_state: {learner_state}{context_block}

输出 JSON 格式：
{"action": "counterexample", "reason": "用户理解了基本关系，但无法识别概念边界", "content": "如果你放弃了三个选择，机会成本是不是三个选择加起来？为什么？", "requires_user_response": true}""",
        concept=concept,
        source_text=source_text,
        learner_state=learner_state,
        current_target=current_target or "验证任务",
        mode_hint=mode_hint,
        context_block=context_block,
    )


def learner_state_updater_prompt(
    *,
    source_text: str,
    concept: str,
    intervention: str,
    user_answer: str,
    learner_state: str,
    context: str = "",
) -> str:
    """Update the Learner State from the learner's newest answer. Returns JSON
    validated by :class:`LearnerStateUpdate`."""
    context_block = f"\n\n此前的对话记录：\n{context}" if context else ""
    return _fill(
        """你是 RecallOS 的学习状态更新器。

根据用户对上次干预的最新回答，更新学习者状态。只给出基于这次最新回答的新快照，
不要机械地把历史条目复制进来。理解层级可以在原有基础上提升，也可以保持不变或更低。

理解层级：
- surface：复述定义或关键词
- relationship：解释概念之间的关系
- application：能够将概念用于具体情境
- essence：理解底层逻辑、因果机制或存在原因

next_best_action 是 AI 下一步最值得做的动作：none / hint / analogy / example / counterexample / question。

规则：
1. 不以回答长短判断质量。
2. 正确但机械复述，只能算 surface。
3. 只输出合法 JSON。

输入：
concept: {concept}
source_text: {source_text}
上次干预：{intervention}
user_answer: {user_answer}
learner_state: {learner_state}{context_block}

输出 JSON 格式：
{"understanding_level": "application", "understood": ["理解机会成本与选择有关", "能够识别具体情境中的机会成本"], "uncertain": [], "misconceptions": [], "last_response_quality": "deep", "next_best_action": "none"}""",
        concept=concept,
        source_text=source_text,
        intervention=intervention,
        user_answer=user_answer,
        learner_state=learner_state,
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
        """你是 RecallOS 的学习教练。

用户已经基本理解「{concept}」。请生成一句自然、有吸引力的继续深入邀请，让用户自己选择是否继续。

要求：
1. 必须体现"你已经理解核心"。
2. 暗示继续深入能获得什么价值。
3. 不使用"你想不想"。
4. 不制造压力。
5. 最多 25 个中文字符。
6. 根据 understanding_level 调整表达。
7. 只输出合法 JSON。

输入：
concept: {concept}
understanding_level: {understanding_level}

输出 JSON 格式：
{"offer": "你已经抓住核心，要不要再挖一层，看看它为什么成立？", "options": ["深入", "先到这里"]}""",
        concept=concept,
        understanding_level=understanding_level,
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
    options: list[NonEmptyStr] = Field(default_factory=lambda: ["深入", "先到这里"])


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
    "intervention_decider_prompt",
    "learner_state_updater_prompt",
    "deepening_offer_prompt",
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
]
