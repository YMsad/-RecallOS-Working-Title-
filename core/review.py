"""Review queue — spaced repetition over the concepts the user has learned.

Flow: finish learning -> add_to_review_queue -> next day get_due_reviews ->
start a ReviewSession -> judge answers, update_review_status on pass/fail.
Every AI call is scriptable through an injected :class:`DeepSeekClient`.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from core.client import DeepSeekClient
from core.database import (
    get_concept,
    get_concepts_due_review,
    get_daily_summaries_for_concept,
    save_review_log,
    update_concept,
)
from core.models import MASTERY_LEARNING, MASTERY_UNCLEAR, MASTERY_UNDERSTOOD
from core.prompts import (
    CheckAnswerResult,
    build_messages,
    check_answer_prompt,
    review_question_prompt,
    validate_response,
)

logger = logging.getLogger(__name__)

MAX_REVIEW_ATTEMPTS = 3
JUDGE_TEMPERATURE = 0.0
OTHER_TEMPERATURE = 0.3


def add_to_review_queue(concept_id: int) -> bool:
    """Schedule a concept for its first review tomorrow. Returns True if updated."""
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    return update_concept(concept_id, next_review_date=tomorrow)


def get_due_reviews(today: str | None = None) -> list[dict[str, Any]]:
    """Return all concepts due for review today or earlier."""
    return get_concepts_due_review(today)


def update_review_status(concept_id: int, passed: bool) -> bool:
    """Advance the review stage on pass; push back to tomorrow on fail."""
    concept = get_concept(concept_id)
    if concept is None:
        return False
    if passed:
        stage = int(concept.get("review_stage") or 0) + 1
        count = int(concept.get("review_count") or 0) + 1
        # Passed again: push out a little further (2 days, then 4, capped at 7).
        gap_days = min(7, 1 << stage)
        next_review = (date.today() + timedelta(days=gap_days)).isoformat()
        update_concept(
            concept_id,
            mastery=MASTERY_UNDERSTOOD,
            review_stage=stage,
            review_count=count,
            next_review_date=next_review,
        )
        return True
    count = int(concept.get("review_count") or 0) + 1
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    update_concept(
        concept_id,
        mastery=MASTERY_UNCLEAR,
        review_stage=0,
        review_count=count,
        next_review_date=tomorrow,
    )
    return True


class ReviewSession:
    """One review session for a single concept, using the day's review question."""

    def __init__(
        self,
        concept_id: int,
        *,
        client: DeepSeekClient | None = None,
    ) -> None:
        self.concept_id = concept_id
        self.client = client or DeepSeekClient()
        concept = get_concept(concept_id)
        if concept is None:
            raise ValueError(f"concept {concept_id} does not exist")
        self.title = concept["title"]
        self.source_text = concept.get("source_text") or ""
        self._question: str | None = None
        self.attempts = 0
        self.phase: str = "reviewing"
        self.needs_relearn: bool = False
        self.last_result: dict[str, Any] | None = None

    def start(self) -> str:
        """Return the first review question: the tomorrow hook, or a fresh one."""
        if self.phase != "reviewing":
            raise ValueError("review already finished")
        hook = self._tomorrow_hook()
        if hook:
            self._question = hook
        else:
            prompt = review_question_prompt(
                title=self.title, source_text=self.source_text
            )
            reply = self.client.chat(
                build_messages(prompt), temperature=OTHER_TEMPERATURE
            )
            self._question = reply.strip()
        return self._question

    def next_question(self) -> str | None:
        return self._question

    def submit_answer(self, answer: str) -> dict[str, Any]:
        """Judge a review answer; returns pass/fail plus feedback."""
        if self.phase != "reviewing" or self._question is None:
            raise ValueError("submit_answer can only be called during review")
        is_last = self.attempts >= MAX_REVIEW_ATTEMPTS - 1
        prompt = check_answer_prompt(
            title=self.title,
            source_text=self.source_text,
            question=self._question,
            answer=answer,
            is_last_attempt=is_last,
        )
        reply = self.client.chat(build_messages(prompt), temperature=JUDGE_TEMPERATURE)
        result: CheckAnswerResult = validate_response(reply, CheckAnswerResult)
        self.attempts += 1
        passed = bool(result.is_correct)
        save_review_log(
            self.concept_id,
            self._question,
            answer,
            passed,
        )

        if passed:
            update_review_status(self.concept_id, passed=True)
            self.phase = "finished"
            self.last_result = {
                "passed": True,
                "feedback": result.feedback,
                "attempts": self.attempts,
            }
            return self.last_result

        if self.attempts >= MAX_REVIEW_ATTEMPTS:
            update_review_status(self.concept_id, passed=False)
            update_concept(self.concept_id, mastery=MASTERY_LEARNING)
            self.needs_relearn = True
            self.phase = "finished"
            self.last_result = {
                "passed": False,
                "feedback": result.feedback,
                "attempts": self.attempts,
                "needs_relearn": True,
            }
            return self.last_result

        # Wrong but attempts remain: refresh the question from the concept.
        self.last_result = {
            "passed": False,
            "feedback": result.feedback,
            "attempts": self.attempts,
        }
        return self.last_result

    def _tomorrow_hook(self) -> str | None:
        summaries = get_daily_summaries_for_concept(self.concept_id)
        for s in summaries:
            if s.get("tomorrow_hook"):
                return s["tomorrow_hook"]
        return None
