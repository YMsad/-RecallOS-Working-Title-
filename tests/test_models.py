"""Tests for the Pydantic models."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from core import (
    MASTERY_LEARNING,
    MASTERY_UNDERSTOOD,
    Concept,
    Connection,
    DailySummary,
    QARecord,
    Setting,
)


# --------------------------------------------------------------------- Concept

def test_concept_requires_title() -> None:
    with pytest.raises(ValidationError):
        Concept()


def test_concept_strips_whitespace() -> None:
    assert Concept(title="  opportunity cost  ").title == "opportunity cost"


def test_concept_empty_title_rejected() -> None:
    with pytest.raises(ValidationError):
        Concept(title="   ")


def test_concept_default_mastery() -> None:
    assert Concept(title="opportunity cost").mastery == MASTERY_LEARNING


def test_concept_valid_mastery() -> None:
    assert Concept(title="opportunity cost", mastery=MASTERY_UNDERSTOOD).mastery == "Understood"


def test_concept_invalid_mastery_rejected() -> None:
    with pytest.raises(ValidationError):
        Concept(title="opportunity cost", mastery="sure")


def test_concept_builds_from_db_row() -> None:
    concept = Concept.model_validate(
        {
            "id": 1,
            "title": "opportunity cost",
            "user_definition": "the value you gave up",
            "source_text": "source",
            "mastery": "Understood",
            "created_at": "2026-08-02 10:00:00",
            "updated_at": "2026-08-02 10:00:00",
        }
    )
    assert concept.id == 1
    assert concept.created_at == datetime(2026, 8, 2, 10, 0, 0)


# -------------------------------------------------------------------- QARecord

def test_qa_requires_concept_id_and_question() -> None:
    with pytest.raises(ValidationError):
        QARecord(concept_id=1)
    with pytest.raises(ValidationError):
        QARecord(question="question")


def test_qa_bool_coercion_from_db() -> None:
    qa = QARecord.model_validate(
        {
            "concept_id": 1,
            "question": "Q?",
            "user_answer": "A",
            "is_correct": 1,
            "hint_used": 0,
        }
    )
    assert qa.is_correct is True
    assert qa.hint_used is False


# ---------------------------------------------------------------- Connection

def test_connection_normalizes_order() -> None:
    conn = Connection(concept_a_id=5, concept_b_id=3, relation_text="related")
    assert conn.concept_a_id == 3
    assert conn.concept_b_id == 5


def test_connection_keeps_sorted_order() -> None:
    conn = Connection(concept_a_id=2, concept_b_id=4, relation_text="related")
    assert (conn.concept_a_id, conn.concept_b_id) == (2, 4)


def test_connection_rejects_self_link() -> None:
    with pytest.raises(ValidationError):
        Connection(concept_a_id=3, concept_b_id=3, relation_text="self-link")


def test_connection_requires_relation_text() -> None:
    with pytest.raises(ValidationError):
        Connection(concept_a_id=1, concept_b_id=2)


# ------------------------------------------------------------- DailySummary

def test_daily_summary_date_defaults_to_today() -> None:
    summary = DailySummary(breakthrough_text="I finally understood")
    assert summary.date == date.today()


def test_daily_summary_parses_date_string() -> None:
    summary = DailySummary.model_validate(
        {"date": "2026-08-02", "breakthrough_text": "understood"}
    )
    assert summary.date == date(2026, 8, 2)


def test_daily_summary_requires_breakthrough() -> None:
    with pytest.raises(ValidationError):
        DailySummary()


# -------------------------------------------------------------------- Setting

def test_setting_key_required_and_stripped() -> None:
    assert Setting(key="  streak  ", value="3").key == "streak"
    with pytest.raises(ValidationError):
        Setting(key="  ")


def test_setting_value_default_empty() -> None:
    assert Setting(key="streak").value == ""
