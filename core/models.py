"""Pydantic models mirroring the five database tables.

- ``Concept``      -> concepts
- ``QARecord``     -> qa_records
- ``Connection``   -> connections   (normalises a < b, rejects self-links)
- ``DailySummary`` -> daily_summaries (date defaults to today)
- ``Setting``      -> settings
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

# ------------------------------------------------------------------ constants

MASTERY_LEARNING = "学习中"
MASTERY_UNCLEAR = "模糊"
MASTERY_UNDERSTOOD = "搞懂了"
MASTERY_VALUES: tuple[str, ...] = (MASTERY_LEARNING, MASTERY_UNCLEAR, MASTERY_UNDERSTOOD)

Mastery = Literal["学习中", "模糊", "搞懂了"]

# ----------------------------------------------------------------- type aliases

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OptionalStr = Annotated[str | None, StringConstraints(strip_whitespace=True)]


class RecallBaseModel(BaseModel):
    """Base for all models — allows building from ORM objects / DB rows."""

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------- models

class Concept(RecallBaseModel):
    id: int | None = None
    title: NonEmptyStr
    user_definition: OptionalStr = None
    source_text: OptionalStr = None
    mastery: Mastery = MASTERY_LEARNING
    created_at: datetime | None = None
    updated_at: datetime | None = None


class QARecord(RecallBaseModel):
    id: int | None = None
    concept_id: int
    question: NonEmptyStr
    user_answer: OptionalStr = None
    is_correct: bool = False
    hint_used: bool = False
    asked_at: datetime | None = None


class Connection(RecallBaseModel):
    id: int | None = None
    concept_a_id: int
    concept_b_id: int
    relation_text: NonEmptyStr
    is_user_edited: bool = False
    created_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_and_normalize(self) -> "Connection":
        if self.concept_a_id == self.concept_b_id:
            raise ValueError("concept_a_id and concept_b_id must be different")
        if self.concept_a_id > self.concept_b_id:
            self.concept_a_id, self.concept_b_id = self.concept_b_id, self.concept_a_id
        return self


class DailySummary(RecallBaseModel):
    id: int | None = None
    date: date_type = Field(default_factory=date_type.today)
    concept_id: int | None = None
    breakthrough_text: NonEmptyStr
    tomorrow_hook: OptionalStr = None
    created_at: datetime | None = None


class Setting(RecallBaseModel):
    key: NonEmptyStr
    value: str = ""
    updated_at: datetime | None = None


__all__ = [
    "MASTERY_LEARNING",
    "MASTERY_UNCLEAR",
    "MASTERY_UNDERSTOOD",
    "MASTERY_VALUES",
    "Mastery",
    "NonEmptyStr",
    "OptionalStr",
    "Concept",
    "QARecord",
    "Connection",
    "DailySummary",
    "Setting",
]
