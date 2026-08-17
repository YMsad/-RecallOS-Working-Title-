"""Learning session — drives the four-layer Socratic flow end to end.

Flow: start -> learning (layers 1-4) -> connections -> finish.
Every AI call is scriptable through an injected :class:`DeepSeekClient`
(e.g. built on ``httpx.MockTransport``) for deterministic tests.
"""

from __future__ import annotations

import json
import logging
import random
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
from core.learner_state import LearnerState
from core.models import MASTERY_UNCLEAR, MASTERY_UNDERSTOOD
from core.review import add_to_review_queue
from core.prompts import (
    DEEPER_QUESTION_ORDER,
    CheckAnswerResult,
    ConnectionSuggestion,
    DeepeningOffer,
    Intervention,
    LearnerStateAnalysis,
    LearnerStateUpdate,
    SummaryResult,
    ValidationTask,
    angle_shift_prompt,
    build_messages,
    check_answer_prompt,
    connections_prompt,
    deepening_offer_prompt,
    explain_prompt,
    intervention_decider_prompt,
    learner_state_analyzer_prompt,
    learner_state_updater_prompt,
    opening_question_prompt,
    question_prompt,
    reference_answer_prompt,
    simplify_explanation_prompt,
    simplify_question_prompt,
    summary_prompt,
    validate_response,
    validate_response_list,
    validation_task_prompt,
    warmup_prompt,
    _legacy_deeper_question_prompt,
)

logger = logging.getLogger(__name__)

MAX_LAYER = 4
QUESTION_TEMPERATURE = 0.7
JUDGE_TEMPERATURE = 0.0
OTHER_TEMPERATURE = 0.3
VALIDATION_MAX_ATTEMPTS = 3

# V0.3.1 — 用户信号输入
LEARNING_GOALS = {
    "understand": "🧠 理解概念",
    "connect": "🔗 建立联系",
    "apply": "🛠 能实际应用",
    "exam": "🎓 为考试掌握",
}
CONFIDENCE_PROMPT_RATE = 0.3  # 验证前偶尔让用户预测一次能否解释清楚

_INTERVENTION_ICONS = {
    "hint": "💡",
    "example": "🌰",
    "analogy": "🎭",
    "counterexample": "⚠️",
    "question": "🤔",
    "rephrase": "🔁",
    "none": "✅",
}


def _intervention_message(intervention: Any) -> str:
    """Bubble text shown to the learner for a decided intervention."""
    icon = _INTERVENTION_ICONS.get(intervention.get("action", ""), "💡")
    return f"{icon} {intervention['content']}"


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
        learning_goal: str = "understand",
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
        self.learning_goal = (
            learning_goal if learning_goal in LEARNING_GOALS else "understand"
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

        # V0.3.0 — Learning Loop v2：Learner State 驱动的动态循环
        self.learner_state = LearnerState()
        self.validation_kind: str | None = None
        self.validation_difficulty: int | None = None
        self._current_intervention: dict[str, Any] | None = None
        self._offer: dict[str, Any] | None = None

        # V0.3.1 — 用户信号输入（全部可选，不阻塞学习流程）
        self.reading_signals: list[dict[str, Any]] = []
        self.stuck_points: list[str] = []
        self.confidence_predictions: list[dict[str, Any]] = []
        self.intervention_feedback_list: list[dict[str, Any]] = []
        self._ask_confidence_this_round: bool = False
        self._intervention_feedback_given: bool = False

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
        """Design one concrete validation task that checks real understanding
        (not rote recall). Enters the validation stage.
        """
        cid = self._require_started()
        logger.info(
            "[DEBUG] start_validation 开始：concept=%s id=%s",
            self.title,
            cid,
        )
        prompt = validation_task_prompt(
            source_text=self.source_text,
            concept=self.title,
            text_type=self.text_type or "",
        )
        reply = self._chat_or_raise(prompt, OTHER_TEMPERATURE, "设计验证任务")
        task = self._parse_or_raise(reply, ValidationTask, "验证任务")
        logger.info("[DEBUG] 验证任务设计完成：task=%r", task.task)
        self.validation_task = task.task
        self.validation_kind = task.type
        self.validation_difficulty = task.difficulty
        self.validation_target = None
        self.validation_attempts = 0
        self.validation_passed = False
        self.needs_relearning = False
        self.learner_state = LearnerState(learning_goal=self.learning_goal)
        self._current_intervention = None
        self._offer = None
        self._ask_confidence_this_round = random.random() < CONFIDENCE_PROMPT_RATE
        self._intervention_feedback_given = False
        self.stage = "validation"
        self._persist_new_flow()
        logger.info("Validation started for concept %s (id=%s)", self.title, cid)
        return self.validation_task

    # ----------------------------------------------- V0.3.1 user signals

    def record_reading_signal(self, kind: str, position: int) -> None:
        """Record a light reading-time understanding marker.

        ``kind`` is one of ``confused`` / ``match`` / ``clear``. Position is the
        0-based index of the paragraph the user marked.
        """
        self.reading_signals.append({"kind": kind, "position": int(position)})
        self._persist_new_flow()

    def record_stuck_point(self, text: str) -> None:
        """Record a concrete stuck point; it feeds into LearnerState.uncertain."""
        text = text.strip()
        if not text:
            return
        self.stuck_points.append(text)
        if text not in self.learner_state.uncertain:
            self.learner_state.uncertain.append(text)
        self._persist_new_flow()

    def should_ask_confidence(self) -> bool:
        """Whether the occasionally-appearing confidence question should show now."""
        return self._ask_confidence_this_round

    def feedback_pending(self) -> bool:
        """Whether the current intervention still awaits the learner's feedback."""
        return (
            self._current_intervention is not None
            and not self._intervention_feedback_given
        )

    def record_confidence_prediction(self, prediction: str) -> None:
        """Record the learner's self-assessment before explaining."""
        self.confidence_predictions.append(
            {"attempt": self.validation_attempts + 1, "prediction": prediction}
        )
        self._ask_confidence_this_round = False
        self._persist_new_flow()

    def record_intervention_feedback(
        self, feedback: str, *, action: str | None = None
    ) -> None:
        """Record what the learner thought of the latest intervention.

        ``feedback`` is ``clear`` (清楚多了) or ``unclear`` (还是有点懵).
        """
        entry: dict[str, Any] = {
            "action": action
            or (
                self._current_intervention.get("action", "")
                if self._current_intervention
                else ""
            ),
            "feedback": feedback,
        }
        self.intervention_feedback_list.append(entry)
        if self.learner_state.intervention_history:
            self.learner_state.intervention_history[-1]["feedback"] = feedback
        else:
            self.learner_state.intervention_history.append(dict(entry))
        self._intervention_feedback_given = True
        self._persist_new_flow()

    def signals_payload(self) -> dict[str, Any]:
        """Compact JSON-friend data used to persist/restore the optional signals."""
        return {
            "reading_signals": list(self.reading_signals),
            "stuck_points": list(self.stuck_points),
            "confidence": list(self.confidence_predictions),
            "intervention_feedback": list(self.intervention_feedback_list),
        }

    def restore_signals(self, payload: dict[str, Any] | None) -> None:
        """Hydrate the optional signals from a previously persisted payload."""
        if not payload:
            return
        self.reading_signals = [
            dict(x) for x in payload.get("reading_signals") or []
        ]
        self.stuck_points = [str(x) for x in payload.get("stuck_points") or []]
        self.confidence_predictions = [
            dict(x) for x in payload.get("confidence") or []
        ]
        self.intervention_feedback_list = [
            dict(x) for x in payload.get("intervention_feedback") or []
        ]

    def submit_validation(self, answer: str) -> dict[str, Any]:
        """Analyse the learner's closed-book answer into a Learner State.

        - No gap → validation passes, move to the deepen offer (stage "offer").
        - Gap → decide the minimal intervention and move to stage
          "intervention" (or finish if no valuable intervention remains).
        """
        cid = self._require_started()
        print(f"[RecallOS][submit_validation] 被调用 cid={cid} stage={self.stage!r} answer_len={len(answer)}", flush=True)
        if self.stage != "validation":
            raise SessionError("submit_validation can only be called during validation")

        analysis = self._analyze_learner(answer)
        print(f"[RecallOS][submit_validation] JSON解析成功 level={analysis.understanding_level!r} understood={analysis.understood} uncertain={analysis.uncertain} misconceptions={analysis.misconceptions}", flush=True)
        # 元认知校准：把这次的实际理解层级回填到最近一次信心预测上
        if self.confidence_predictions:
            self.confidence_predictions[-1]["actual_level"] = analysis.understanding_level
        self.validation_history.append(
            {
                "answer": answer,
                "understanding_level": analysis.understanding_level,
                "last_response_quality": analysis.last_response_quality,
                "understood": analysis.understood,
                "uncertain": analysis.uncertain,
                "misconceptions": analysis.misconceptions,
            }
        )
        self.validation_attempts += 1
        self._persist_new_flow()

        if not self.learner_state.has_gap():
            self.validation_passed = True
            self.validation_attempts = 0
            self.stage = "offer"
            self._persist_new_flow()
            logger.info(
                "Validation passed for concept %s (level=%s)",
                self.title,
                self.learner_state.understanding_level,
            )
            return {
                "stage": "offer",
                "understanding_level": self.learner_state.understanding_level,
                "quality": self.learner_state.last_response_quality,
            }

        intervention = self._decide_intervention(mode="validation")
        if intervention is None:
            return self._finish_new_flow()
        return self._apply_intervention(intervention)

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
            _legacy_deeper_question_prompt(
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
        """V0.3.0 — AI 判断不可用时，用户可手动「跳过」验证。

        仅用于临时放行：把本轮验证视为通过，进入深入选择阶段，不等待 AI。
        """
        cid = self._require_started()
        if self.stage != "validation":
            raise SessionError(
                "force_validation_pass can only be called during validation"
            )
        self.validation_passed = True
        self.validation_attempts = 0
        self.stage = "offer"
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

    # ------------------------------------------- V0.3.0 Learning Loop v2
    # Learner State 驱动的动态循环：验证 → （修复/深化）→ 深入选择 → 动态结束。

    def offer_deepening(self) -> dict[str, Any]:
        """Ask the learner (in their own voice) whether to keep going deeper
        after validation passed. Returns {"offer": str, "options": [...]}."""
        cid = self._require_started()
        if self.stage != "offer":
            raise SessionError("offer_deepening can only be called at offer stage")
        reply = self._chat_or_raise(
            deepening_offer_prompt(
                concept=self.title,
                understanding_level=self.learner_state.understanding_level,
            ),
            OTHER_TEMPERATURE,
            "深化邀请",
        )
        offer = self._parse_or_raise(reply, DeepeningOffer, "深化邀请")
        self._offer = {
            "offer": offer.offer,
            "options": [str(o) for o in offer.options],
        }
        logger.info("Deepen offer generated for concept %s (id=%s)", self.title, cid)
        return dict(self._offer)

    def choose_deepening(self, go: bool) -> dict[str, Any]:
        """Apply the learner's decision after the deepen offer.

        - ``go=False`` → finish the session now.
        - ``go=True`` → target the next valuable gap with a minimal
          intervention (stage "intervention").
        """
        cid = self._require_started()
        if self.stage != "offer":
            raise SessionError("choose_deepening can only be called at offer stage")
        if not go:
            logger.info("Learner chose to stop after validation (concept=%s)", self.title)
            return self._finish_new_flow()
        intervention = self._decide_intervention(mode="deepening")
        if intervention is None:
            return self._finish_new_flow()
        return self._apply_intervention(intervention)

    def submit_intervention_answer(self, answer: str) -> dict[str, Any]:
        """Update the Learner State from the learner's answer to the current
        minimal intervention, then decide what to do next.

        - No valuable gap left → finish.
        - Gap remains → the next minimal intervention (stage stays
          "intervention").
        """
        cid = self._require_started()
        if self.stage != "intervention" or self._current_intervention is None:
            raise SessionError(
                "no active intervention; call submit_validation/choose_deepening first"
            )
        current = self._current_intervention
        update = self._update_learner(answer)
        self.deeper_history.append(
            {
                "question": current["content"],
                "answer": answer,
                "action": current["action"],
                "understanding_level": update.understanding_level,
                "last_response_quality": update.last_response_quality,
                "understood": update.understood,
                "uncertain": update.uncertain,
                "misconceptions": update.misconceptions,
                "next_best_action": update.next_best_action,
            }
        )
        self.current_deeper_index += 1
        self._current_intervention = None
        self._persist_new_flow()
        logger.info(
            "Intervention answered for concept %s (id=%s, level=%s)",
            self.title,
            cid,
            self.learner_state.understanding_level,
        )

        if self.learner_state.should_stop():
            return self._finish_new_flow()
        next_intervention = self._decide_intervention(mode="deepening")
        if next_intervention is None:
            return self._finish_new_flow()
        return self._apply_intervention(next_intervention)

    def current_intervention(self) -> dict[str, Any] | None:
        """The minimal intervention currently on screen, or None."""
        return self._current_intervention

    def next_intervention(self, *, mode: str = "deepening") -> dict[str, Any]:
        """Decide and surface the next minimal intervention (used on resume /
        auto-refresh when none is on screen)."""
        cid = self._require_started()
        if self._current_intervention is not None:
            return {
                "stage": "intervention",
                "bubble": _intervention_message(self._current_intervention),
            }
        intervention = self._decide_intervention(mode=mode)
        if intervention is None:
            return self._finish_new_flow()
        return self._apply_intervention(intervention)

    def _apply_intervention(self, intervention: dict[str, Any]) -> dict[str, Any]:
        """Surface a decided minimal intervention on screen.

        If the intervention is a closing note that does not need a user
        response (``requires_user_response`` false), finish the flow with it
        instead of waiting for an answer.
        """
        if not intervention.get("requires_user_response", True):
            return self._finish_new_flow(intervention)
        self._current_intervention = intervention
        self._intervention_feedback_given = False  # 新一轮干预：先收集反馈
        self.learner_state.intervention_history.append(
            {
                "action": intervention.get("action", ""),
                "reason": intervention.get("reason", ""),
                "feedback": "",
            }
        )
        self.stage = "intervention"
        self._persist_new_flow()
        return {
            "stage": "intervention",
            "bubble": _intervention_message(intervention),
        }

    # ------------------------------------------------------------- v2 internals

    def _analyze_learner(self, answer: str) -> LearnerStateAnalysis:
        # 卡住点在进入分析前先落入 uncertain，并一并喂给分析器，避免被覆盖丢失
        for sp in self.stuck_points:
            if sp not in self.learner_state.uncertain:
                self.learner_state.uncertain.append(sp)
        last_confidence = (
            self.confidence_predictions[-1]["prediction"]
            if self.confidence_predictions
            else ""
        )
        reply = self._chat_or_raise(
            learner_state_analyzer_prompt(
                source_text=self.source_text,
                concept=self.title,
                task=self.validation_task or "用自己的话解释这个概念",
                user_answer=answer,
                context=self._format_learner_context(),
                learning_goal=self.learning_goal,
                stuck_points="\n".join(self.stuck_points),
                confidence_prediction=str(last_confidence),
            ),
            JUDGE_TEMPERATURE,
            "分析学习者状态",
        )
        print(f"[RecallOS][_analyze_learner] AI原始回复(前500字符)：{reply[:500]!r}", flush=True)
        analysis = self._parse_or_raise(reply, LearnerStateAnalysis, "学习者状态分析")
        self.learner_state.update_from_analysis(analysis.model_dump())
        for sp in self.stuck_points:  # 保留用户自述的卡住点
            if sp not in self.learner_state.uncertain:
                self.learner_state.uncertain.append(sp)
        return analysis

    def _update_learner(self, answer: str) -> LearnerStateUpdate:
        reply = self._chat_or_raise(
            learner_state_updater_prompt(
                source_text=self.source_text,
                concept=self.title,
                intervention=self._current_intervention["content"]
                if self._current_intervention
                else "",
                user_answer=answer,
                learner_state=json.dumps(
                    self.learner_state.to_dict(), ensure_ascii=False
                ),
                context=self._format_learner_context(),
                learning_goal=self.learning_goal,
            ),
            JUDGE_TEMPERATURE,
            "更新学习者状态",
        )
        update = self._parse_or_raise(reply, LearnerStateUpdate, "学习者状态更新")
        self.learner_state.update_from_analysis(update.model_dump())
        return update

    def _decide_intervention(
        self, *, mode: str
    ) -> dict[str, Any] | None:
        reply = self._chat_or_raise(
            intervention_decider_prompt(
                source_text=self.source_text,
                concept=self.title,
                learner_state=json.dumps(
                    self.learner_state.to_dict(), ensure_ascii=False
                ),
                current_target=self.validation_task or "用自己的话解释这个概念",
                mode=mode,
                context=self._format_learner_context(),
                intervention_history=json.dumps(
                    [
                        {k: h.get(k) for k in ("action", "feedback")}
                        for h in self.learner_state.intervention_history
                        if h.get("feedback")
                    ],
                    ensure_ascii=False,
                ),
            ),
            QUESTION_TEMPERATURE,
            "决定最小干预",
        )
        inter = self._parse_or_raise(reply, Intervention, "干预决策")
        if inter.action == "none":
            # 没有值得继续的干预：若给了收尾内容，则作为最终提示后结束
            if inter.content:
                return {
                    "action": "none",
                    "reason": inter.reason or "",
                    "content": inter.content,
                    "requires_user_response": False,
                }
            return None
        return {
            "action": inter.action,
            "reason": inter.reason or "",
            "content": inter.content,
            "requires_user_response": inter.requires_user_response,
        }

    def _finish_new_flow(
        self, final_intervention: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Finish the new flow: complete/connections + add to review queue."""
        cid = self._require_started()
        self._current_intervention = None
        self.stage = "complete"
        self.phase = "connections"
        self._persist_new_flow()
        add_to_review_queue(cid)
        logger.info(
            "New flow finished for concept %s (id=%s, level=%s)",
            self.title,
            cid,
            self.learner_state.understanding_level,
        )
        result: dict[str, Any] = {"stage": "complete"}
        if final_intervention and final_intervention.get("content"):
            result["final_note"] = _intervention_message(final_intervention)
        return result

    def _format_learner_context(self) -> str:
        """Compact context for the analyzer/decider/updater prompts."""
        lines: list[str] = []
        for i, qa in enumerate(self.qa_history[-6:], 1):
            lines.append(f"{i}. Q: {qa['question']}")
            lines.append(f"   A: {qa['answer']}")
        for i, v in enumerate(self.validation_history, 1):
            lines.append(
                f"验证{i}: {v['answer']}（层级 {v.get('understanding_level')}）"
            )
        for i, d in enumerate(self.deeper_history, 1):
            lines.append(
                f"干预{i}: {d['question']} → {d['answer']}（层级 {d.get('understanding_level')}）"
            )
        return "\n".join(lines)

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
            validation_kind=self.validation_kind,
            validation_difficulty=self.validation_difficulty,
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
            learning_goal=self.learning_goal,
            intervention_feedback=(
                json.dumps(self.intervention_feedback_list, ensure_ascii=False)
                if self.intervention_feedback_list
                else None
            ),
            signals=(
                json.dumps(self.signals_payload(), ensure_ascii=False)
                if (
                    self.reading_signals
                    or self.stuck_points
                    or self.confidence_predictions
                )
                else None
            ),
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


def _load_json_dict(raw: Any) -> dict[str, Any]:
    """Parse a JSON-object column value; return {} for empty/invalid content."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


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
    session.learning_goal = (
        concept.get("learning_goal")
        if concept.get("learning_goal") in LEARNING_GOALS
        else "understand"
    )
    feedback_payload = _load_json_list(concept.get("intervention_feedback"))
    if feedback_payload:
        session.intervention_feedback_list = [dict(x) for x in feedback_payload]
    session.restore_signals(_load_json_dict(concept.get("signals")))

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
        session.validation_kind = concept.get("validation_kind")
        session.validation_difficulty = int(concept.get("validation_difficulty") or 2)
        session.validation_passed = bool(concept.get("validation_passed"))
        session.validation_attempts = int(concept.get("validation_attempts") or 0)
        session.needs_relearning = bool(concept.get("needs_relearning"))
        session.validation_history = _load_json_list(concept.get("validation_history"))
        session.deeper_questions = _load_json_list(concept.get("deeper_questions"))
        session.deeper_history = _load_json_list(concept.get("deeper_answers"))
        session.current_deeper_index = int(concept.get("deeper_index") or 0)
        # Learning Loop v2：从最近的回答快照重建 Learner State
        snapshot: dict[str, Any] | None = None
        for entry in reversed(session.deeper_history):
            if "understanding_level" in entry:
                snapshot = entry
                break
        if snapshot is None:
            for entry in reversed(session.validation_history):
                if "understanding_level" in entry:
                    snapshot = entry
                    break
        session.learner_state = LearnerState.from_dict(snapshot)
        session.learner_state.learning_goal = session.learning_goal
        if session.stage == "complete":
            session.phase = "connections"
        elif session.stage == "intervention":
            # 屏幕上那道最小干预在回答前不会落库；恢复后由 UI 自动决策下一条。
            session._current_intervention = None
        elif session.stage == "deepening":
            # 兼容旧数据（重构前的固定深化阶段）：恢复屏幕上的深化问题
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
