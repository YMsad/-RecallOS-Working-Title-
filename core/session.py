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
    build_messages,
    check_answer_prompt,
    connections_prompt,
    question_prompt,
    reference_answer_prompt,
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
    ) -> None:
        self.title = title.strip()
        self.source_text = source_text
        self.client = client or DeepSeekClient()
        self.max_consecutive_failures = max_consecutive_failures

        # State
        self.concept_id: int | None = None
        self.layer: int = 0
        self.phase: str = "learning"
        self.consecutive_failures: int = 0
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
        self._current_question = self._generate_question(layer=1)
        logger.info("Session started for concept %s (id=%s)", self.title, self.concept_id)
        return self._current_question

    def next_question(self) -> str | None:
        """Return the current question, or None once learning has finished."""
        return self._current_question

    def submit_answer(self, answer: str) -> dict[str, Any]:
        """Judge the answer; advance the layer on success, offer hints on failure."""
        cid = self._require_started()
        if self.phase != "learning" or self._current_question is None:
            raise SessionError("submit_answer can only be called during learning")

        question = self._current_question
        judgement = self._judge_answer(answer)
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

        if self.layer > MAX_LAYER:
            self.phase = "connections"
        result["is_done"] = self.phase != "learning"
        return result

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
        )
        reply = self.client.chat(build_messages(prompt), temperature=QUESTION_TEMPERATURE)
        return reply.strip()

    def _judge_answer(self, answer: str) -> dict[str, str | bool | None]:
        prompt = check_answer_prompt(
            title=self.title,
            source_text=self.source_text,
            question=self._current_question or "",
            answer=answer,
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
