"""Tests for the SQLite persistence layer (five tables)."""

from __future__ import annotations

import sqlite3

import pytest

from core import database


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Point the database at a throwaway file before every test."""
    database.configure(tmp_path / "test.db")
    yield


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def test_init_creates_five_tables() -> None:
    with database._get_conn() as conn:
        assert table_names(conn) == {
            "concepts",
            "qa_records",
            "connections",
            "daily_summaries",
            "settings",
        }


def test_save_and_get_concept() -> None:
    cid = database.save_concept("机会成本", "为了得到它必须放弃的东西")
    row = database.get_concept(cid)
    assert row is not None
    assert row["title"] == "机会成本"
    assert row["source_text"] == "为了得到它必须放弃的东西"
    assert row["mastery"] == database.MASTERY_LEARNING


def test_get_missing_concept_returns_none() -> None:
    assert database.get_concept(999) is None


def test_update_concept() -> None:
    cid = database.save_concept("沉没成本")
    assert database.update_concept(cid, mastery=database.MASTERY_UNDERSTOOD,
                                   user_definition="收不回来的投入")
    row = database.get_concept(cid)
    assert row["mastery"] == database.MASTERY_UNDERSTOOD
    assert row["user_definition"] == "收不回来的投入"


def test_update_concept_validates_mastery() -> None:
    cid = database.save_concept("边际效用")
    with pytest.raises(ValueError):
        database.update_concept(cid, mastery="sure")


def test_get_recent_concepts_orders_by_updated() -> None:
    first = database.save_concept("稀缺")
    second = database.save_concept("机会成本")
    database.update_concept(first, user_definition="X")  # touch first
    recent = database.get_recent_concepts(limit=5)
    assert [c["id"] for c in recent] == [first, second]


def test_save_qa_and_history() -> None:
    cid = database.save_concept("机会成本")
    qa1 = database.save_qa(cid, "它关注过去还是未来？", "未来", True)
    qa2 = database.save_qa(cid, "和沉没成本的区别？", "", False, hint_used=True)
    history = database.get_qa_history(cid)
    assert [h["id"] for h in history] == [qa1, qa2]
    assert history[1]["is_correct"] == 0
    assert history[1]["hint_used"] == 1


def test_save_connection_normalizes_order() -> None:
    a = database.save_concept("机会成本")
    b = database.save_concept("沉没成本")
    database.save_connection(a, b, "一个看未来，一个看过去")
    # reversed insertion must not duplicate
    database.save_connection(b, a, "一个看未来，一个看过去")
    conns = database.get_connections(a)
    assert len(conns) == 1
    assert conns[0]["concept_a_id"] == min(a, b)
    assert conns[0]["concept_b_id"] == max(a, b)
    assert conns[0]["concept_a_title"] in {"机会成本", "沉没成本"}


def test_get_connections_returns_both_directions() -> None:
    a = database.save_concept("机会成本")
    b = database.save_concept("沉没成本")
    c = database.save_concept("边际效用")
    database.save_connection(a, b, "都是选择概念")
    database.save_connection(a, c, "都与价值有关")
    conns = database.get_connections(a)
    assert len(conns) == 2


def test_cascade_delete_removes_qa_and_connections() -> None:
    a = database.save_concept("机会成本")
    b = database.save_concept("沉没成本")
    database.save_qa(a, "Q?", "A", True)
    database.save_connection(a, b, "关联")
    assert database.delete_concept(a) is True
    assert database.get_qa_history(a) == []
    assert database.get_connections(a) == []
    assert database.get_all_connections() == []


def test_foreign_key_enforced() -> None:
    with pytest.raises(sqlite3.IntegrityError):
        database.save_qa(999, "Q?", "A", True)


def test_connection_check_constraint_rejects_self_and_reversed() -> None:
    a = database.save_concept("机会成本")
    b = database.save_concept("沉没成本")
    with database._get_conn() as conn:
        # a == b violates CHECK (concept_a_id != concept_b_id)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO connections (concept_a_id, concept_b_id, relation_text) "
                "VALUES (?, ?, ?)",
                (a, a, "自连"),
            )
        # reversed order violates CHECK (concept_a_id < concept_b_id)
        high, low = (a, b) if a > b else (b, a)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO connections (concept_a_id, concept_b_id, relation_text) "
                "VALUES (?, ?, ?)",
                (high, low, "乱序"),
            )


def test_daily_summary() -> None:
    cid = database.save_concept("供需平衡")
    database.save_daily_summary(cid, "我终于搞懂了供需曲线",
                                tomorrow_hook="边际效用与机会成本的关系？")
    today = database.get_today_summary()
    assert today is not None
    assert today["concept_id"] == cid
    assert today["breakthrough_text"] == "我终于搞懂了供需曲线"
    assert database.get_daily_summaries(limit=5)[0]["tomorrow_hook"]


def test_daily_summary_none_when_nothing_saved() -> None:
    assert database.get_today_summary() is None


def test_daily_summaries_for_concept() -> None:
    c1 = database.save_concept("机会成本")
    c2 = database.save_concept("沉没成本")
    database.save_daily_summary(c1, "搞懂了机会成本")
    database.save_daily_summary(c2, "搞懂了沉没成本", date="2026-08-01")
    summaries = database.get_daily_summaries_for_concept(c1)
    assert len(summaries) == 1
    assert summaries[0]["breakthrough_text"] == "搞懂了机会成本"
    assert database.get_daily_summaries_for_concept(c2)[0]["date"] == "2026-08-01"


def test_settings_get_set() -> None:
    assert database.get_setting("streak") is None
    assert database.get_setting("streak", default="0") == "0"
    database.set_setting("streak", "3")
    assert database.get_setting("streak") == "3"
    database.set_setting("streak", "4")
    assert database.get_setting("streak") == "4"
