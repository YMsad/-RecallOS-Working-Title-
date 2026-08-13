"""SQLite persistence layer — five tables: concepts, qa_records, connections,
daily_summaries, settings.

Schema follows 技术文档.md (V0.1) with the Chinese table name "追问记录"
corrected to ``qa_records``. All CRUD uses short-lived connections; point the
database at a test path with :func:`configure`.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator

from core.config import PROJECT_ROOT
from core.models import (
    MASTERY_LEARNING,
    MASTERY_UNCLEAR,
    MASTERY_UNDERSTOOD,
    MASTERY_VALUES,
)

DB_PATH: Path = PROJECT_ROOT / "data" / "recallos.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    user_definition TEXT,
    source_text TEXT,
    mastery TEXT NOT NULL DEFAULT '学习中',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    next_review_date DATE,
    review_stage INTEGER NOT NULL DEFAULT 0,
    review_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS qa_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    user_answer TEXT,
    is_correct INTEGER NOT NULL DEFAULT 0,
    hint_used INTEGER NOT NULL DEFAULT 0,
    asked_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_a_id INTEGER NOT NULL,
    concept_b_id INTEGER NOT NULL,
    relation_text TEXT NOT NULL,
    is_user_edited INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (concept_a_id) REFERENCES concepts(id) ON DELETE CASCADE,
    FOREIGN KEY (concept_b_id) REFERENCES concepts(id) ON DELETE CASCADE,
    UNIQUE(concept_a_id, concept_b_id),
    CHECK (concept_a_id != concept_b_id),
    CHECK (concept_a_id < concept_b_id)
);

CREATE TABLE IF NOT EXISTS review_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL,
    review_date DATE NOT NULL DEFAULT CURRENT_DATE,
    question TEXT NOT NULL,
    user_answer TEXT,
    passed INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS daily_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL DEFAULT CURRENT_DATE,
    concept_id INTEGER,
    breakthrough_text TEXT,
    tomorrow_hook TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_concepts_updated ON concepts(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_concept ON qa_records(concept_id);
CREATE INDEX IF NOT EXISTS idx_connections_a ON connections(concept_a_id);
CREATE INDEX IF NOT EXISTS idx_connections_b ON connections(concept_b_id);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summaries(date);
"""


def configure(db_path: str | Path) -> None:
    """Point the database at a new path and initialise it (used by tests)."""
    global DB_PATH
    DB_PATH = Path(db_path)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    init_db()


@contextmanager
def _get_conn() -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection with safety pragmas; auto commit/rollback."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the data directory, tables and indexes. Idempotent."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add V0.2.2 columns to pre-existing concepts tables (idempotent)."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(concepts)")}
    for col, ddl in (
        ("next_review_date", "next_review_date DATE"),
        ("review_stage", "review_stage INTEGER NOT NULL DEFAULT 0"),
        ("review_count", "review_count INTEGER NOT NULL DEFAULT 0"),
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE concepts ADD COLUMN {ddl}")


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


# --------------------------------------------------------------------- concepts

def save_concept(title: str, source_text: str | None = None) -> int:
    """Save a new concept and return its id."""
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO concepts (title, source_text) VALUES (?, ?)",
            (title, source_text),
        )
        return int(cur.lastrowid)


def get_concept(concept_id: int) -> dict[str, Any] | None:
    """Return a concept by id, or None."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM concepts WHERE id = ?", (concept_id,)
        ).fetchone()
        return dict(row) if row else None


def get_all_concepts() -> list[dict[str, Any]]:
    """Return all concepts, most recently updated first."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM concepts ORDER BY updated_at DESC").fetchall()
        return _rows_to_dicts(rows)


def get_recent_concepts(limit: int = 10) -> list[dict[str, Any]]:
    """Return the N most recently updated concepts."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM concepts ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return _rows_to_dicts(rows)


def update_concept(
    concept_id: int,
    *,
    title: str | None = None,
    user_definition: str | None = None,
    source_text: str | None = None,
    mastery: str | None = None,
    next_review_date: str | None = None,
    review_stage: int | None = None,
    review_count: int | None = None,
) -> bool:
    """Update any subset of a concept's fields. Returns True if a row changed."""
    if mastery is not None and mastery not in MASTERY_VALUES:
        raise ValueError(f"mastery must be one of {MASTERY_VALUES}, got {mastery!r}")
    fields: dict[str, Any] = {}
    if title is not None:
        fields["title"] = title
    if user_definition is not None:
        fields["user_definition"] = user_definition
    if source_text is not None:
        fields["source_text"] = source_text
    if mastery is not None:
        fields["mastery"] = mastery
    if next_review_date is not None:
        fields["next_review_date"] = next_review_date
    if review_stage is not None:
        fields["review_stage"] = review_stage
    if review_count is not None:
        fields["review_count"] = review_count
    if not fields:
        return True
    set_clause = ", ".join(f"{key} = ?" for key in fields)
    params = [*fields.values(), concept_id]
    with _get_conn() as conn:
        cur = conn.execute(
            f"UPDATE concepts SET {set_clause}, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            params,
        )
        return cur.rowcount > 0


def delete_concept(concept_id: int) -> bool:
    """Delete a concept; cascades to qa_records and connections. Returns True if deleted."""
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM concepts WHERE id = ?", (concept_id,))
        return cur.rowcount > 0


# ----------------------------------------------------------------- qa_records

def save_qa(
    concept_id: int,
    question: str,
    user_answer: str | None,
    is_correct: bool,
    hint_used: bool = False,
) -> int:
    """Save one Q&A turn and return its id."""
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO qa_records (concept_id, question, user_answer, is_correct, hint_used) "
            "VALUES (?, ?, ?, ?, ?)",
            (concept_id, question, user_answer, int(is_correct), int(hint_used)),
        )
        return int(cur.lastrowid)


def get_qa_history(concept_id: int) -> list[dict[str, Any]]:
    """Return all Q&A turns for a concept, oldest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM qa_records WHERE concept_id = ? ORDER BY asked_at",
            (concept_id,),
        ).fetchall()
        return _rows_to_dicts(rows)


# ---------------------------------------------------------------- connections

def save_connection(
    concept_a_id: int,
    concept_b_id: int,
    relation_text: str,
    is_user_edited: bool = False,
) -> int:
    """Save a connection, normalising (a, b) so a < b. Upserts on conflict. Returns its id."""
    a, b = sorted((concept_a_id, concept_b_id))
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO connections "
            "(concept_a_id, concept_b_id, relation_text, is_user_edited) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(concept_a_id, concept_b_id) DO UPDATE SET "
            "relation_text = excluded.relation_text, "
            "is_user_edited = excluded.is_user_edited",
            (a, b, relation_text, int(is_user_edited)),
        )
        row = conn.execute(
            "SELECT id FROM connections WHERE concept_a_id = ? AND concept_b_id = ?",
            (a, b),
        ).fetchone()
        return int(row["id"])


def get_connections(concept_id: int) -> list[dict[str, Any]]:
    """Return connections involving a concept, including both titles."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.concept_a_id, c.concept_b_id, c.relation_text,
                   c.is_user_edited, c.created_at,
                   ca.title AS concept_a_title, cb.title AS concept_b_title
            FROM connections c
            JOIN concepts ca ON ca.id = c.concept_a_id
            JOIN concepts cb ON cb.id = c.concept_b_id
            WHERE c.concept_a_id = ? OR c.concept_b_id = ?
            ORDER BY c.created_at
            """,
            (concept_id, concept_id),
        ).fetchall()
        return _rows_to_dicts(rows)


def get_all_connections() -> list[dict[str, Any]]:
    """Return all connections with both concept titles."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.concept_a_id, c.concept_b_id, c.relation_text,
                   c.is_user_edited, c.created_at,
                   ca.title AS concept_a_title, cb.title AS concept_b_title
            FROM connections c
            JOIN concepts ca ON ca.id = c.concept_a_id
            JOIN concepts cb ON cb.id = c.concept_b_id
            ORDER BY c.created_at
            """
        ).fetchall()
        return _rows_to_dicts(rows)


# --------------------------------------------------------------------- reviews

def get_concepts_due_review(today: str | None = None) -> list[dict[str, Any]]:
    """Return concepts whose next_review_date is today or overdue."""
    if today is None:
        today = date.today().isoformat()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM concepts WHERE next_review_date IS NOT NULL "
            "AND next_review_date <= ? ORDER BY next_review_date",
            (today,),
        ).fetchall()
        return _rows_to_dicts(rows)


def get_all_review_logs(limit: int = 100) -> list[dict[str, Any]]:
    """Return the most recent review log entries, newest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM review_log ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return _rows_to_dicts(rows)


def get_review_logs(concept_id: int) -> list[dict[str, Any]]:
    """Return review log entries for one concept, oldest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM review_log WHERE concept_id = ? ORDER BY created_at",
            (concept_id,),
        ).fetchall()
        return _rows_to_dicts(rows)


def save_review_log(
    concept_id: int,
    question: str,
    user_answer: str | None,
    passed: bool,
    review_date: str | None = None,
) -> int:
    """Record one review attempt and return its id."""
    if review_date is None:
        with _get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO review_log (concept_id, question, user_answer, passed) "
                "VALUES (?, ?, ?, ?)",
                (concept_id, question, user_answer, int(passed)),
            )
            return int(cur.lastrowid)
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO review_log (concept_id, review_date, question, user_answer, passed) "
            "VALUES (?, ?, ?, ?, ?)",
            (concept_id, review_date, question, user_answer, int(passed)),
        )
        return int(cur.lastrowid)


# ----------------------------------------------------------- daily_summaries

def save_daily_summary(
    concept_id: int | None,
    breakthrough_text: str,
    tomorrow_hook: str | None = None,
    date: str | None = None,
) -> int:
    """Save a daily summary and return its id."""
    if date is None:
        with _get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO daily_summaries (concept_id, breakthrough_text, tomorrow_hook) "
                "VALUES (?, ?, ?)",
                (concept_id, breakthrough_text, tomorrow_hook),
            )
            return int(cur.lastrowid)
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO daily_summaries (date, concept_id, breakthrough_text, tomorrow_hook) "
            "VALUES (?, ?, ?, ?)",
            (date, concept_id, breakthrough_text, tomorrow_hook),
        )
        return int(cur.lastrowid)


def get_today_summary() -> dict[str, Any] | None:
    """Return today's summary, or None."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM daily_summaries WHERE date = date('now') "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def get_daily_summaries(limit: int = 30) -> list[dict[str, Any]]:
    """Return recent daily summaries, newest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM daily_summaries ORDER BY date DESC, created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return _rows_to_dicts(rows)


def get_daily_summaries_for_concept(concept_id: int) -> list[dict[str, Any]]:
    """Return all daily summaries for one concept, newest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM daily_summaries WHERE concept_id = ? "
            "ORDER BY date DESC, created_at DESC",
            (concept_id,),
        ).fetchall()
        return _rows_to_dicts(rows)


# ------------------------------------------------------------------- settings

def get_setting(key: str, default: str | None = None) -> str | None:
    """Return a setting value, or ``default`` if unset."""
    with _get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    """Set (or replace) a setting value."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP)",
            (key, value),
        )
