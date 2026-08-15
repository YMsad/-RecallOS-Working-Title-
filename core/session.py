"""Learning session — drives the four-layer Socratic flow end to end.

Flow: start -> learning (layers 1-4) -> connections -> finish.
Every AI call is scriptable through an injected :class:`DeepSeekClient`
(e.g. built on ``httpx.MockTransport``) for deterministic tests.
"""

from __future__ import annotations

import json
import logging
import warnings
from functools import wraps
from typing import Any, Callable

from pydantic import ValidationError

from core.client import DeepSeekClient, DeepSeekError
from core.database import (
    get_all_concepts,
    get_concept,
    get_qa_history,
    save_concept,
    save_connection,
    save_daily_summary,
    save_qa,
    update_concept,
)
from core.models import MASTERY_UNCLEAR, MASTERY_UNDERSTOOD
from core.review import add_to_review_queue
from core.prompts import (
    DEEPER_QUESTION_ORDER,
    CheckAnswerResult,
    ConnectionSuggestion,
    SummaryResult,
    ValidateAnswerResult,
    ValidationTask,
    angle_shift_prompt,
    build_messages,
    check_answer_prompt,
    connections_prompt,
    deeper_question_prompt,
    explain_prompt,
    opening_question_prompt,
    question_prompt,
    reference_answer_prompt,
    simplify_explanation_prompt,
    simplify_question_prompt,
    summary_prompt,
    validate_answer_prompt,
    validate_response,
    validate_response_list,
    validation_task_prompt,
    warmup_prompt,
)

logger = logging.getLogger(__name__)

MAX_LAYER = 4
QUESTION_TEMPERATURE = 0.7
JUDGE_TEMPERATURE = 0.0
OTHER_TEMPERATURE = 0.3
VALIDATION_MAX_ATTEMPTS = 3


def deprecated(message: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a method/function as deprecated (V0.3.0 keeps old code intact and
    only warns; the callers will be migrated in later steps)."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                message or f"{func.__name__} is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


class SessionError(Exception):
    """Raised when a session method is called out of order."""


def warmup_concept(
    title: str,
    source_text: str,
    *,
    client: DeepSeekClient | None = None,
) -> str:
    """Give a 1-2 sentence plain-language intro for a concept, without needing a
    full session. Used by the home page's 预热 button before a session starts."""
    prompt = warmup_prompt(title=title, source_text=source_text)
    c = client or DeepSeekClient()
    reply = c.chat(build_messages(prompt), temperature=OTHER_TEMPERATURE)
    return reply.strip()


class LearningSession:
    """One Socratic learning session for a single concept."""

    def __init__(
        self,
        title: str,
        source_text: str,
        *,
        client: DeepSeekClient | None = None,
        max_consecutive_failures: int = 3,
        mode: str = "beginner",
        level: str = "zero",
        interest: str = "simple",
    ) -> None:
        self.title = title.strip()
        self.source_text = source_text
        self.client = client or DeepSeekClient()
        self.max_consecutive_failures = max_consecutive_failures
        self.mode = mode if mode in ("beginner", "advanced") else "beginner"
        self.level = level if level in ("zero", "some", "familiar") else "zero"
        self.interest = (
            interest if interest in ("simple", "deep", "example") else "simple"
        )

        # State
        self.concept_id: int | None = None
        self.layer: int = 0
        self.phase: str = "learning"
        self.consecutive_failures: int = 0
        self.explain_used: bool = False
        self.marked_uncertain: bool = False
        self.qa_history: list[dict[str, Any]] = []
        self._current_question: str | None = None
        self.recommended_connections: list[ConnectionSuggestion] = []
        self.summary: SummaryResult | None = None

        # V0.3.0 — 底层逻辑重构新增状态（旧的 phase/层推进逻辑保持不变，仍可用）
        self.stage: str = "learning"
        self.text_type: str | None = None
        self.validation_task: str | None = None
        self.validation_target: str | None = None
        self.validation_attempts: int = 0
        self.validation_passed: bool = False
        self.validation_history: list[dict[str, Any]] = []
        self.needs_relearning: bool = False
        self.deeper_questions: list[str] = []
        self.current_deeper_index: int = 0
        self._current_deeper_question: str | None = None
        self.deeper_history: list[dict[str, str]] = []
        self._last_explanation: str | None = None

        # V0.3.0 — 流程标记：''new''＝新流程（阅读→验证→深化），''legacy''＝旧四层追问
        self.flow: str = "legacy"

    # ------------------------------------------------------------------ flow

    def start(self) -> str:
        """Save the concept and generate the first question. Returns the question."""
        self.concept_id = save_concept(self.title, self.source_text)
        self.phase = "learning"
        self.layer = 1
        self._current_question = self._generate_opening_question()
        logger.info("Session started for concept %s (id=%s)", self.title, self.concept_id)
        return self._current_question

    def begin(self) -> int:
        """V0.3.0 — start the new flow without the old opening question: only
        persist the concept and enter the reading stage. Returns the concept id.
        """
        if self.concept_id is not None:
            return self.concept_id
        self.concept_id = save_concept(self.title, self.source_text)
        self.phase = "learning"
        self.stage = "reading"
        self._persist_new_flow()
        logger.info(
            "Session begun (V0.3.0 flow) for concept %s (id=%s)",
            self.title,
            self.concept_id,
        )
        return self.concept_id

    def warmup(self) -> str:
        """Give a 1-2 sentence plain-language intro (zero-basis pre-warm), or '' if
        already known. Does not require the session to be started."""
        if self.mode != "beginner":
            return ""
        return warmup_concept(self.title, self.source_text, client=self.client)

    def next_question(self) -> str | None:
        """Return the current question, or None once learning has finished."""
        return self._current_question

    def submit_answer(self, answer: str) -> dict[str, Any]:
        """Judge the answer; advance the layer on success, escalate support on failure."""
        cid = self._require_started()
        if self.phase != "learning" or self._current_question is None:
            raise SessionError("submit_answer can only be called during learning")

        question = self._current_question
        is_last_attempt = self.consecutive_failures >= self.max_consecutive_failures - 1
        judgement = self._judge_answer(answer, is_last_attempt=is_last_attempt)
        self.qa_history.append(
            {
                "question": question,
                "answer": answer,
                "is_correct": judgement["correct"],
                "hint": judgement["hint"],
            }
        )
        save_qa(
            cid,
            question,
            answer,
            is_correct=judgement["correct"],
            hint_used=judgement["hint"] is not None,
        )

        result: dict[str, Any] = {
            "question": question,
            "correct": judgement["correct"],
            "feedback": judgement["feedback"],
            "hint": judgement["hint"],
            "reference": None,
            "mastery": None,
            "is_done": False,
            "simplified": False,
            "angle_shift": False,
            "explain_used": self.explain_used,
        }

        if judgement["correct"]:
            self.consecutive_failures = 0
            self._advance_layer()
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                reference = self._ask_reference()
                update_concept(cid, mastery=MASTERY_UNCLEAR)
                self.marked_uncertain = True
                self.consecutive_failures = 0
                result["reference"] = reference
                result["mastery"] = MASTERY_UNCLEAR
                self._advance_layer()
            elif self.consecutive_failures == 1:
                # 降维：第一次答错后换更简单、更具体的问题
                self._current_question = self._simplify_question(question)
                result["simplified"] = True
            elif self.consecutive_failures == 2:
                # 换角度：第二次答错后，先问要不要换个角度
                self._current_question = self._angle_shift(question)
                result["angle_shift"] = True

        if self.layer > MAX_LAYER:
            self.phase = "connections"
            # V0.2.3 — 学习完成即加入复习队列（无论是否走完总结等后续流程）
            add_to_review_queue(cid)
            logger.info("概念已加入复习队列: %s", cid)
        result["is_done"] = self.phase != "learning"
        return result

    def ask_for_angle_switch(self) -> str:
        """Explicitly offer '换个角度理解？' without waiting for another wrong answer.

        Returns the angle-shifted question.
        """
        cid = self._require_started()
        if self.phase != "learning" or self._current_question is None:
            raise SessionError("ask_for_angle_switch can only be called during learning")
        self._current_question = self._angle_shift(self._current_question)
        self.consecutive_failures = 0
        logger.info("Angle switch offered for concept %s (id=%s)", self.title, cid)
        return self._current_question

    def explain(self) -> str:
        """Explain the current question in plain language. Returns the explanation."""
        cid = self._require_started()
        if self.phase != "learning" or self._current_question is None:
            raise SessionError("explain can only be called during learning")
        prompt = explain_prompt(
            title=self.title,
            source_text=self.source_text,
            question=self._current_question,
        )
        reply = self.client.chat(build_messages(prompt), temperature=OTHER_TEMPERATURE)
        self.explain_used = True
        self.consecutive_failures = 0
        logger.info("Explanation given for concept %s (id=%s)", self.title, cid)
        return reply.strip()

    def get_connections(self) -> list[ConnectionSuggestion]:
        """Recommend and persist knowledge connections. Called after learning."""
        cid = self._require_started()
        if self.phase != "connections":
            raise SessionError("get_connections can only be called after learning finishes")

        all_concepts = get_all_concepts()
        titles = [c["title"] for c in all_concepts if c["id"] != cid]
        prompt = connections_prompt(
            title=self.title, source_text=self.source_text, all_concepts=titles
        )
        reply = self.client.chat(build_messages(prompt), temperature=OTHER_TEMPERATURE)
        suggestions = validate_response_list(reply, ConnectionSuggestion)

        saved = 0
        for s in suggestions:
            target = next(
                (c for c in all_concepts if c["title"] == s.concept_title), None
            )
            if target is not None:
                save_connection(cid, target["id"], s.relation_text)
                saved += 1
        logger.info("Saved %d/%d suggested connections", saved, len(suggestions))
        self.recommended_connections = suggestions
        return suggestions

    def finish(self, user_definition: str = "") -> SummaryResult:
        """Generate the daily summary, persist it, and close the session."""
        cid = self._require_started()
        if self.phase not in ("connections", "finished"):
            raise SessionError("finish can only be called after learning finishes")

        prompt = summary_prompt(
            title=self.title,
            user_definition=user_definition,
            qa_history=self._format_history(),
        )
        reply = self.client.chat(build_messages(prompt), temperature=OTHER_TEMPERATURE)
        self.summary = validate_response(reply, SummaryResult)

        save_daily_summary(cid, self.summary.breakthrough, self.summary.tomorrow_hook)
        mastery = MASTERY_UNCLEAR if self.marked_uncertain else MASTERY_UNDERSTOOD
        update_concept(
            cid,
            user_definition=user_definition or None,
            mastery=mastery,
        )
        self.phase = "finished"
        logger.info("Session finished for concept %s (mastery=%s)", self.title, mastery)
        return self.summary

    # ----------------------------------------------------- V0.3.0 validation

    def start_validation(self) -> str:
        """Begin the validation stage: design one concrete validation task that
        checks real understanding. Returns the task text.
        """
        cid = self._require_started()
        logger.info(
            "[DEBUG] start_validation 开始：concept=%s id=%s（拼接验证任务 prompt）",
            self.title,
            cid,
        )
        prompt = validation_task_prompt(
            title=self.title,
            source_text=self.source_text,
            qa_history=self._format_history(),
        )
        reply = self._chat_or_raise(prompt, OTHER_TEMPERATURE, "设计验证任务")
        task = self._parse_or_raise(reply, ValidationTask, "验证任务")
        logger.info("[DEBUG] 验证任务设计完成：task=%r", task.task)
        self.validation_task = task.task
        self.validation_target = task.target
        self.validation_attempts = 0
        self.validation_passed = False
        self.needs_relearning = False
        self.stage = "validation"
        self._persist_new_flow()
        logger.info("Validation started for concept %s (id=%s)", self.title, cid)
        return self.validation_task

    def submit_validation(self, answer: str) -> dict[str, Any]:
        """Judge the validation-task answer. After 3 consecutive failures the
        concept is marked as「需要重新学习」.
        """
        cid = self._require_started()
        if self.stage != "validation" or self.validation_task is None:
            raise SessionError("submit_validation can only be called during validation")
        if self.validation_target is None:
            raise SessionError("validation target is missing; call start_validation() first")

        attempts_left = VALIDATION_MAX_ATTEMPTS - self.validation_attempts
        reply = self._chat_or_raise(
            validate_answer_prompt(
                title=self.title,
                task=self.validation_task,
                target=self.validation_target,
                answer=answer,
                attempts_left=attempts_left,
            ),
            JUDGE_TEMPERATURE,
            "判断验证作答",
        )
        result = self._parse_or_raise(reply, ValidateAnswerResult, "验证判定")

        if result.is_correct:
            self.validation_passed = True
            self.validation_attempts = 0
            # 验证通过 → 进入深化追问阶段
            self.stage = "deepening"
        else:
            self.validation_attempts += 1
            if self.validation_attempts >= VALIDATION_MAX_ATTEMPTS:
                self.needs_relearning = True
                self.stage = "relearn"
                logger.info(
                    "Validation failed %d times for concept %s (id=%s) -> needs relearning",
                    VALIDATION_MAX_ATTEMPTS,
                    self.title,
                    cid,
                )

        self.validation_history.append(
            {
                "answer": answer,
                "passed": bool(result.is_correct),
                "feedback": result.feedback,
                "missing": result.missing,
            }
        )
        self._persist_new_flow()

        return {
            "passed": self.validation_passed,
            "feedback": result.feedback,
            "missing": result.missing,
            "attempts_left": max(
                0, VALIDATION_MAX_ATTEMPTS - self.validation_attempts
            ),
            "needs_relearning": self.needs_relearning,
            "stage": self.stage,
        }

    def ask_simplify(self) -> str:
        """Give a one-notch-simpler plain-language explanation of the concept.
        Does not change the current stage.
        """
        cid = self._require_started()
        reply = self._chat_or_raise(
            simplify_explanation_prompt(
                title=self.title,
                source_text=self.source_text,
                explanation=self._last_explanation or "",
            ),
            OTHER_TEMPERATURE,
            "大白话解释",
        )
        self._last_explanation = reply.strip()
        logger.info("Simplified explanation given for concept %s (id=%s)", self.title, cid)
        return self._last_explanation

    # ------------------------------------------------------- V0.3.0 deepening

    def next_deeper_question(self) -> str | None:
        """Produce the next deeper-probing question (再验证 -> 联系 -> 反事实 ->
        行动 -> 第一性原理), or None once all five have been asked.
        """
        cid = self._require_started()
        if self.current_deeper_index >= len(DEEPER_QUESTION_ORDER):
            self._current_deeper_question = None
            # 深化追问全部问完 → 新流程学习完成，进入连接/总结阶段并入复习队列
            self.stage = "complete"
            self.phase = "connections"
            self._persist_new_flow()
            add_to_review_queue(cid)
            return None

        qtype = DEEPER_QUESTION_ORDER[self.current_deeper_index]
        reply = self._chat_or_raise(
            deeper_question_prompt(
                title=self.title,
                source_text=self.source_text,
                question_type=qtype,
                qa_history=self._format_history(),
            ),
            QUESTION_TEMPERATURE,
            "生成深化追问",
        )
        question = reply.strip()
        self.deeper_questions.append(question)
        self._current_deeper_question = question
        self.current_deeper_index += 1
        self._persist_new_flow()
        logger.info(
            "Deeper question %d/%d (%s) for concept %s (id=%s)",
            self.current_deeper_index,
            len(DEEPER_QUESTION_ORDER),
            qtype,
            self.title,
            cid,
        )
        return question

    def submit_deeper_answer(self, answer: str) -> dict[str, Any]:
        """Record the learner's answer to the current deeper question. Deeper
        questions are open-ended, so no pass/fail is judged here; the exchange
        is kept in :attr:`deeper_history`.
        """
        cid = self._require_started()
        if self._current_deeper_question is None:
            raise SessionError(
                "no deeper question on screen; call next_deeper_question() first"
            )
        self.deeper_history.append(
            {
                "question": self._current_deeper_question,
                "answer": answer,
            }
        )
        self._persist_new_flow()
        logger.info("Deeper answer recorded for concept %s (id=%s)", self.title, cid)
        return {
            "question": self._current_deeper_question,
            "recorded": True,
            "deeper_asked": self.current_deeper_index,
        }

    def force_validation_pass(self) -> None:
        """V0.3.0 — AI 判断不可用时，用户可手动「跳过」验证并进入深化阶段。

        仅用于临时放行：把本轮验证视为通过，进入 deepening，不等待 AI。
        """
        cid = self._require_started()
        if self.stage != "validation":
            raise SessionError(
                "force_validation_pass can only be called during validation"
            )
        self.validation_passed = True
        self.validation_attempts = 0
        self.stage = "deepening"
        self.validation_history.append(
            {
                "answer": "(由用户选择跳过，未经过 AI 判断)",
                "passed": True,
                "feedback": "用户手动跳过（AI 不可用）",
                "missing": "",
            }
        )
        self._persist_new_flow()
        logger.info(
            "Validation manually skipped (pass) for concept %s (id=%s)", self.title, cid
        )

    def finish_deepening(self) -> None:
        """V0.3.0 — 提前结束深化阶段（AI 无法继续生成问题时，用户手动跳过）。

        效果与自然问完 5 问一致：mark 为 complete/connections 并入复习队列。
        """
        cid = self._require_started()
        self._current_deeper_question = None
        self.current_deeper_index = len(DEEPER_QUESTION_ORDER)
        self.stage = "complete"
        self.phase = "connections"
        self._persist_new_flow()
        add_to_review_queue(cid)
        logger.info("Deepening manually finished early for concept %s (id=%s)", self.title, cid)

    def _chat_or_raise(self, prompt: str, temperature: float, what: str) -> str:
        """Send one AI request with explicit logging around the call.

        Failures are logged with context then re-raised as ``DeepSeekError`` so
        the UI layer can show a retry-able error prompt instead of hanging.
        """
        cid = self._require_started()
        snippet = " ".join(prompt.split())[:60]
        logger.info(
            "AI 调用开始[%s]: concept=%s id=%s 提问=%r", what, self.title, cid, snippet
        )
        try:
            reply = self.client.chat(build_messages(prompt), temperature=temperature)
        except DeepSeekError as exc:
            logger.error(
                "AI 调用失败[%s]: concept=%s id=%s —— %s", what, self.title, cid, exc
            )
            raise
        logger.info(
            "AI 调用完成[%s]: concept=%s id=%s 回复长度=%d",
            what,
            self.title,
            cid,
            len(reply),
        )
        return reply

    def _parse_or_raise(self, reply: str, model: Any, what: str) -> Any:
        """Validate AI JSON output with a friendly failure.

        A malformed reply must become a retry-able ``SessionError`` the UI can
        show, not an uncaught pydantic/ValueError that crashes the page.
        """
        try:
            return validate_response(reply, model)
        except (ValueError, ValidationError) as exc:
            logger.error(
                "AI 返回解析失败[%s]: %s —— 回复前 200 字=%r",
                what,
                exc,
                reply[:200],
            )
            raise SessionError(f"AI 返回的「{what}」格式不正确，请重试。") from exc

    # --------------------------------------------------------------- internals

    def _persist_new_flow(self) -> None:
        """V0.3.0 — 把新流程（阅读→验证→深化）的会话状态写入 concepts 表，
        这样「继续学习」可以从数据库恢复中断的会话。"""
        if self.concept_id is None:
            return
        update_concept(
            self.concept_id,
            stage=self.stage,
            validation_type=self.text_type,
            validation_task=self.validation_task,
            validation_target=self.validation_target,
            validation_passed=self.validation_passed,
            validation_attempts=self.validation_attempts,
            needs_relearning=self.needs_relearning,
            validation_history=(
                json.dumps(self.validation_history, ensure_ascii=False)
                if self.validation_history
                else None
            ),
            deeper_questions=(
                json.dumps(self.deeper_questions, ensure_ascii=False)
                if self.deeper_questions
                else None
            ),
            deeper_answers=(
                json.dumps(self.deeper_history, ensure_ascii=False)
                if self.deeper_history
                else None
            ),
            deeper_index=self.current_deeper_index,
        )

    def _require_started(self) -> int:
        if self.concept_id is None:
            raise SessionError("session has not been started; call start() first")
        return self.concept_id

    def _generate_opening_question(self) -> str:
        prompt = opening_question_prompt(
            title=self.title,
            source_text=self.source_text,
            level=self.level,
            interest=self.interest,
            mode=self.mode,
        )
        reply = self.client.chat(build_messages(prompt), temperature=QUESTION_TEMPERATURE)
        return reply.strip()

    @deprecated(
        "LearningSession._generate_question is deprecated since V0.3.0; "
        "use next_deeper_question() and the new flow instead"
    )
    def _generate_question(self, layer: int) -> str:
        related = None
        if layer == 4:
            related = [
                c["title"]
                for c in get_all_concepts()
                if c["id"] != self.concept_id
            ]
        prompt = question_prompt(
            layer,
            title=self.title,
            source_text=self.source_text,
            qa_history=self._format_history(),
            related_concepts=related,
            mode=self.mode,
            cognitive_contrast=self.mode == "beginner" and layer in (2, 3),
            model=self._route_model(layer),
        )
        reply = self.client.chat(build_messages(prompt), temperature=QUESTION_TEMPERATURE)
        return reply.strip()

    @deprecated(
        "LearningSession._route_model is deprecated since V0.3.0; "
        "the V0.3.0 flow routes through DEEPER_QUESTION_ORDER instead"
    )
    def _route_model(self, layer: int) -> str | None:
        """V0.2.1 — 思维模型自动路由：根据用户表现自动选择追问模型，不加选择负担。
        V0.2.3 — 增加趣味性：第 3 层（反事实）统一换用黄金圈，避免全程同一个模型。

        - 第 4 层（连接）→ 类比：用「这就像你之前学的 X…」引导连接
        - 零基础模式 → 场景化：让抽象概念落地到生活场景
          （第 3 层改用黄金圈，保持新鲜感）
        - 有基础模式：
          - 已连续答对 → 第 1-2 层第一性原理：往更深处拆；
            第 3 层换黄金圈：回到 Why，夯实根本
          - 出现连错 / 换过角度 → 黄金圈：回到 Why，重新夯实根本
        - 其余情况不注入特定模型（默认苏格拉底四层追问）
        """
        if layer == 4:
            return "analogy"
        if self.mode == "beginner":
            return "golden_circle" if layer == 3 else "scenario"
        if self.mode == "advanced":
            if self.consecutive_failures == 0 and not self.marked_uncertain:
                return "golden_circle" if layer == 3 else "first_principles"
            return "golden_circle"
        return None

    def _simplify_question(self, question: str) -> str:
        prompt = simplify_question_prompt(
            title=self.title,
            source_text=self.source_text,
            question=question,
            mode=self.mode,
        )
        reply = self.client.chat(build_messages(prompt), temperature=QUESTION_TEMPERATURE)
        return reply.strip()

    def _angle_shift(self, question: str) -> str:
        prompt = angle_shift_prompt(
            title=self.title,
            source_text=self.source_text,
            question=question,
            mode=self.mode,
        )
        reply = self.client.chat(build_messages(prompt), temperature=QUESTION_TEMPERATURE)
        return reply.strip()

    def _judge_answer(
        self, answer: str, *, is_last_attempt: bool = False
    ) -> dict[str, str | bool | None]:
        prompt = check_answer_prompt(
            title=self.title,
            source_text=self.source_text,
            question=self._current_question or "",
            answer=answer,
            is_last_attempt=is_last_attempt,
        )
        reply = self.client.chat(build_messages(prompt), temperature=JUDGE_TEMPERATURE)
        result: CheckAnswerResult = validate_response(reply, CheckAnswerResult)
        return {
            "correct": result.is_correct,
            "feedback": result.feedback,
            "hint": result.hint,
        }

    def _ask_reference(self) -> str:
        prompt = reference_answer_prompt(
            title=self.title,
            source_text=self.source_text,
            question=self._current_question or "",
            attempts=self.max_consecutive_failures,
        )
        reply = self.client.chat(build_messages(prompt), temperature=OTHER_TEMPERATURE)
        return reply.strip()

    def _advance_layer(self) -> None:
        self.layer += 1
        if self.layer > MAX_LAYER:
            self._current_question = None
        else:
            self._current_question = self._generate_question(self.layer)

    def _format_history(self) -> str:
        if not self.qa_history:
            return ""
        lines: list[str] = []
        for i, qa in enumerate(self.qa_history, 1):
            hint = f"（提示：{qa['hint']}）" if qa.get("hint") else ""
            lines.append(f"{i}. 问题：{qa['question']}")
            lines.append(f"   回答：{qa['answer']}{hint}")
        return "\n".join(lines)


def _load_json_list(raw: Any) -> list[Any]:
    """Parse a JSON-list column value; return [] for empty/invalid content."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def restore_session(
    concept_id: int, *, client: DeepSeekClient | None = None
) -> LearningSession:
    """V0.3.0 — 从数据库恢复一个已开始的「继续学习」会话。

    - 若概念带有新流程标记（``stage`` 或 ``validation_type`` 非空，即它是在
      V0.3.0 新流程里开始学习的），重建完整的「阅读→验证→深化」状态，
      精确恢复到上次中断的阶段（含验证次数、深化进度、验证与深化回答历史）。
    - 否则退化为旧流程恢复（只重建四层追问的 qa_history 与当前问题）。
    """
    concept = get_concept(concept_id)
    if concept is None:
        raise SessionError(f"concept {concept_id} not found")

    session = LearningSession(
        concept["title"],
        concept.get("source_text") or "",
        client=client,
    )
    session.concept_id = concept_id

    # 通用部分：重建旧流程追问历史（新流程里同样作为补充上下文）
    history = get_qa_history(concept_id)
    for qa in history:
        session.qa_history.append(
            {
                "question": qa["question"],
                "answer": qa.get("user_answer"),
                "is_correct": bool(qa.get("is_correct")),
                "hint": None,
            }
        )
    if history:
        session.layer = min(MAX_LAYER, len(history))
        session._current_question = history[-1]["question"]

    # 新流程部分：带 stage / validation_type 标记 → 恢复完整新流程状态
    stage = concept.get("stage")
    if stage is not None or concept.get("validation_type") is not None:
        session.flow = "new"
        session.stage = stage if stage is not None else "reading"
        session.text_type = concept.get("validation_type")
        session.validation_task = concept.get("validation_task")
        session.validation_target = concept.get("validation_target")
        session.validation_passed = bool(concept.get("validation_passed"))
        session.validation_attempts = int(concept.get("validation_attempts") or 0)
        session.needs_relearning = bool(concept.get("needs_relearning"))
        session.validation_history = _load_json_list(concept.get("validation_history"))
        session.deeper_questions = _load_json_list(concept.get("deeper_questions"))
        session.deeper_history = _load_json_list(concept.get("deeper_answers"))
        session.current_deeper_index = int(concept.get("deeper_index") or 0)
        if session.stage == "complete":
            session.phase = "connections"
        elif session.stage == "deepening":
            # 恢复「屏幕上那道深化问题」：若最后一道已被回答过，则留空让 UI
            # 生成下一道；否则把它还原到屏幕上让用户继续作答。
            last_generated = (
                session.deeper_questions[-1] if session.deeper_questions else None
            )
            answered = (
                session.deeper_history[-1]["question"]
                if session.deeper_history
                else None
            )
            session._current_deeper_question = (
                last_generated if last_generated is not None and answered != last_generated else None
            )
    return session


__all__ = [
    "LearningSession",
    "SessionError",
    "warmup_concept",
    "restore_session",
    "VALIDATION_MAX_ATTEMPTS",
    "MAX_LAYER",
]
