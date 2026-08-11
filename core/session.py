"""Learning session — drives the four-layer Socratic flow end to end.

Flow: start -> learning (layers 1-4) -> connections -> finish.
Every AI call is scriptable through an injected :class:`DeepSeekClient`
(e.g. built on ``httpx.MockTransport``) for deterministic tests.
"""

from __future__ import annotations

import logging
from typing import Any

from core.client import DeepSeekClient
from core.database import (
    get_all_concepts,
    save_concept,
    save_connection,
    save_daily_summary,
    save_qa,
    update_concept,
)
from core.models import MASTERY_UNCLEAR, MASTERY_UNDERSTOOD
from core.prompts import (
    CheckAnswerResult,
    ConnectionSuggestion,
    SummaryResult,
    angle_shift_prompt,
    build_messages,
    check_answer_prompt,
    connections_prompt,
    explain_prompt,
    opening_question_prompt,
    question_prompt,
    reference_answer_prompt,
    simplify_question_prompt,
    summary_prompt,
    validate_response,
    validate_response_list,
)

logger = logging.getLogger(__name__)

MAX_LAYER = 4
QUESTION_TEMPERATURE = 0.7
JUDGE_TEMPERATURE = 0.0
OTHER_TEMPERATURE = 0.3


class SessionError(Exception):
    """Raised when a session method is called out of order."""


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

    # ------------------------------------------------------------------ flow

    def start(self) -> str:
        """Save the concept and generate the first question. Returns the question."""
        self.concept_id = save_concept(self.title, self.source_text)
        self.phase = "learning"
        self.layer = 1
        self._current_question = self._generate_opening_question()
        logger.info("Session started for concept %s (id=%s)", self.title, self.concept_id)
        return self._current_question

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

    # --------------------------------------------------------------- internals

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
        )
        reply = self.client.chat(build_messages(prompt), temperature=QUESTION_TEMPERATURE)
        return reply.strip()

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


__all__ = ["LearningSession", "SessionError"]
