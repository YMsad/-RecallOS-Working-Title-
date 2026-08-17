"""Review queue — spaced repetition over the concepts the user has learned.

Flow
----
1. Finish a learning session -> :func:`add_to_review_queue`  (due tomorrow).
2. On a later day             -> :func:`get_due_reviews`     (today or overdue).
3. Open a review session      -> :class:`ReviewSession`.
4. Judge answers              -> :meth:`ReviewSession.submit_answer`.
5. Schedule the next review   -> :func:`update_review_status`.

The interval grows with every successful pass (2, 4, ... capped at 7 days);
a failed pass resets the schedule back to tomorrow so the concept is re-tested
quickly.  A concept that fails ``MAX_REVIEW_ATTEMPTS`` times in one session is
marked :data:`MASTERY_LEARNING` and flagged for relearning.

Every AI call is scriptable through an injected :class:`DeepSeekClient`
(e.g. built on ``httpx.MockTransport``) for deterministic tests.
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

# A concept must survive this many answers before the interval is allowed to
# grow; failing this many times in one session marks it for relearning.
MAX_REVIEW_ATTEMPTS = 3
# Longest gap between reviews (days). Stage 3 and beyond stay at this value.
MAX_REVIEW_INTERVAL_DAYS = 7
# Judging answers must be deterministic, so we always use greedy decoding.
JUDGE_TEMPERATURE = 0.0
# Question generation can afford to be a little more creative.
OTHER_TEMPERATURE = 0.3


def _today_iso() -> str:
    """Today's date as ISO 8601 — the base for every schedule decision."""
    return date.today().isoformat()


def _days_from_today(days: int) -> str:
    """Date ``days`` from today (negative = in the past), as ISO 8601."""
    return (date.today() + timedelta(days=days)).isoformat()


def _next_review_date(stage: int) -> str:
    """Spacing interval for a given review stage: ``2**stage`` days, capped.

    ``stage`` counts completed passes: stage 1 -> 2 days, 2 -> 4, 3+ -> 7.
    """
    gap_days = min(MAX_REVIEW_INTERVAL_DAYS, 1 << stage)
    return _days_from_today(gap_days)


# ---------------------------------------------------------------- queue

def add_to_review_queue(concept_id: int) -> bool:
    """Schedule a concept for its first review tomorrow. Returns True if updated."""
    return update_concept(concept_id, next_review_date=_days_from_today(1))


def get_due_reviews(today: str | None = None) -> list[dict[str, Any]]:
    """Return all concepts due on ``today`` or earlier, oldest due first."""
    return get_concepts_due_review(today)


def update_review_status(concept_id: int, passed: bool) -> bool:
    """Advance the review schedule after a pass; reset to tomorrow on a fail.

    Pass
        review_stage +1, review_count +1, mastery -> UNDERSTOOD and
        next_review_date pushed out by :func:`_next_review_date`.
    Fail
        review_count +1, review_stage reset to 0, mastery -> UNCLEAR and
        next_review_date -> tomorrow.

    Returns False when the concept does not exist.
    """
    concept = get_concept(concept_id)
    if concept is None:
        return False

    count = int(concept.get("review_count") or 0) + 1
    if passed:
        stage = int(concept.get("review_stage") or 0) + 1
        update_concept(
            concept_id,
            mastery=MASTERY_UNDERSTOOD,
            review_stage=stage,
            review_count=count,
            next_review_date=_next_review_date(stage),
        )
    else:
        update_concept(
            concept_id,
            mastery=MASTERY_UNCLEAR,
            review_stage=0,
            review_count=count,
            next_review_date=_days_from_today(1),
        )
    return True


# --------------------------------------------------------------- session

class ReviewSession:
    """One review session for a single concept.

    The first question is the ``tomorrow_hook`` captured at the end of the
    last learning session (falling back to an AI-generated question).  Each
    answer is judged by the AI; a pass finishes the session immediately,
    while a fail keeps asking until ``MAX_REVIEW_ATTEMPTS`` are exhausted.

    Attributes
    ----------
    phase : str
        ``"reviewing"`` while answers are being accepted, ``"finished"``
        after a pass or after exhausting all attempts.
    needs_relearn : bool
        True when the learner failed ``MAX_REVIEW_ATTEMPTS`` times in a row.
    last_result : dict[str, Any] | None
        Outcome of the most recent :meth:`submit_answer` call, or None.
    """

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

    # ----------------------------------------------------------------- flow

    def start(self) -> str:
        """Return the first question: today's hook if any, else an AI question."""
        if self.phase != "reviewing":
            raise ValueError("review already finished")
        hook = self._tomorrow_hook()
        self._question = hook or self._ask_ai_for_question()
        return self._question

    def next_question(self) -> str | None:
        """Return the current question (None until :meth:`start` is called)."""
        return self._question

    def submit_answer(self, answer: str) -> dict[str, Any]:
        """Judge one answer and return a result dict.

        Result keys: ``passed``, ``feedback``, ``attempts`` and — when the
        learner ran out of attempts — ``needs_relearn``.  Raises ValueError
        when called before :meth:`start` or after the session has finished.
        """
        if self.phase != "reviewing" or self._question is None:
            raise ValueError("submit_answer can only be called during review")

        is_last = self.attempts >= MAX_REVIEW_ATTEMPTS - 1
        result = self._judge(answer, is_last=is_last)
        self.attempts += 1
        passed = bool(result.is_correct)
        save_review_log(self.concept_id, self._question, answer, passed)

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

        # Wrong but attempts remain: give feedback and let the learner retry.
        self.last_result = {
            "passed": False,
            "feedback": result.feedback,
            "attempts": self.attempts,
        }
        return self.last_result

    # ------------------------------------------------------------- helpers

    def _ask_ai_for_question(self) -> str:
        """Generate a fresh review question for this concept."""
        prompt = review_question_prompt(
            title=self.title, source_text=self.source_text
        )
        reply = self.client.chat(
            build_messages(prompt), temperature=OTHER_TEMPERATURE
        )
        return reply.strip()

    def _judge(self, answer: str, *, is_last: bool) -> CheckAnswerResult:
        """Ask the AI to judge an answer, returning the validated result."""
        prompt = check_answer_prompt(
            title=self.title,
            source_text=self.source_text,
            question=self._question or "",
            answer=answer,
            is_last_attempt=is_last,
        )
        reply = self.client.chat(build_messages(prompt), temperature=JUDGE_TEMPERATURE)
        return validate_response(reply, CheckAnswerResult)

    def _tomorrow_hook(self) -> str | None:
        """Return the latest ``tomorrow_hook`` from the previous sessions."""
        summaries = get_daily_summaries_for_concept(self.concept_id)
        for s in summaries:
            if s.get("tomorrow_hook"):
                return s["tomorrow_hook"]
        return None
