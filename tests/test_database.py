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


def test_init_creates_seven_tables() -> None:
    with database._get_conn() as conn:
        assert table_names(conn) == {
            "concepts",
            "qa_records",
            "connections",
            "review_log",
            "daily_summaries",
            "settings",
            "usage_logs",
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


def test_delete_concept_cascade() -> None:
    a = database.save_concept("机会成本")
    b = database.save_concept("沉没成本")
    database.save_qa(a, "它关注过去还是未来？", "未来", True)
    database.save_connection(a, b, "一个看过去一个看未来")
    database.save_review_log(a, "机会成本是什么？", "放弃的价值", True)
    database.save_daily_summary(a, "我终于搞懂了机会成本", "明天再想")
    database.save_usage_log(model="deepseek-chat", total_tokens=10, concept_id=a)

    assert database.delete_concept(a) is True
    assert database.get_concept(a) is None
    assert database.get_qa_history(a) == []
    assert database.get_connections(a) == []
    assert database.get_all_connections() == []
    assert database.get_review_logs(a) == []
    assert database.get_daily_summaries_for_concept(a) == []
    assert database.get_usage_summary()["calls"] == 0
    # 与之建立连接的另一概念不受影响
    assert database.get_concept(b) is not None


def test_delete_concept_not_found() -> None:
    assert database.delete_concept(999999) is False


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


# ----------------------------------------------------------------- usage_logs


def test_save_usage_log_and_summary() -> None:
    log_id = database.save_usage_log(
        model="deepseek-chat",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost=0.001,
        session_id="s-1",
        concept_id=1,
    )
    assert log_id > 0
    summary = database.get_usage_summary()
    assert summary["calls"] == 1
    assert summary["prompt_tokens"] == 100
    assert summary["completion_tokens"] == 50
    assert summary["total_tokens"] == 150
    assert summary["cost"] == 0.001


def test_save_usage_log_defaults_to_zero() -> None:
    database.save_usage_log()
    row = database.get_usage_summary()
    assert row["calls"] == 1
    assert row["total_tokens"] == 0
    assert row["cost"] == 0.0


def test_usage_summary_aggregates_multiple_rows() -> None:
    for _ in range(3):
        database.save_usage_log(
            model="deepseek-chat", prompt_tokens=10, completion_tokens=5,
            total_tokens=15, cost=0.0005,
        )
    summary = database.get_usage_summary()
    assert summary["calls"] == 3
    assert summary["total_tokens"] == 45
    assert round(summary["cost"], 4) == 0.0015


def test_usage_summary_empty_db() -> None:
    summary = database.get_usage_summary()
    assert summary == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                       "total_tokens": 0, "cost": 0.0}


def test_usage_summary_since_filter() -> None:
    database.save_usage_log(model="deepseek-chat", total_tokens=15)
    # A far-past filter should include it; a far-future one should exclude it.
    assert database.get_usage_summary(since="2000-01-01")["calls"] == 1
    assert database.get_usage_summary(since="2999-01-01")["calls"] == 0


def test_usage_trend_groups_by_day() -> None:
    database.save_usage_log(model="deepseek-chat", total_tokens=15, cost=0.001)
    database.save_usage_log(model="deepseek-chat", total_tokens=25, cost=0.002)
    trend = database.get_usage_trend(days=7)
    assert len(trend) == 1
    assert trend[0]["calls"] == 2
    assert trend[0]["total_tokens"] == 40
    assert round(trend[0]["cost"], 4) == 0.003


def test_update_concept_user_signals_columns() -> None:
    """V0.3.1 — learning_goal / intervention_feedback / signals 三列可读写。
    旧库迁移（ALTER TABLE ADD COLUMN）后同样可写。"""
    cid = database.save_concept("机会成本")
    assert database.update_concept(
        cid,
        learning_goal="apply",
        intervention_feedback='[{"action": "hint", "feedback": "clear"}]',
        signals='{"reading_signals": []}',
    )
    row = database.get_concept(cid)
    assert row["learning_goal"] == "apply"
    assert row["intervention_feedback"] == '[{"action": "hint", "feedback": "clear"}]'
    assert row["signals"] == '{"reading_signals": []}'

    # None 参数不覆盖已写值；合法值才会写入
    database.update_concept(cid, mastery=database.MASTERY_UNDERSTOOD)
    row = database.get_concept(cid)
    assert row["learning_goal"] == "apply"


def test_migration_adds_user_signal_columns(tmp_path) -> None:
    """模拟 V0.3.0 老库（无三列）→ init_db 迁移后自动补齐。"""
    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        conn.execute(
            "CREATE TABLE concepts (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "title TEXT NOT NULL, source_text TEXT, mastery TEXT, "
            "question_level INTEGER NOT NULL DEFAULT 1, "
            "user_definition TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, "
            "updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
    database.configure(legacy)  # 切换 DB_PATH 并触发 init_db 迁移
    assert database.DB_PATH == legacy
    with sqlite3.connect(legacy) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(concepts)").fetchall()}
    assert {"learning_goal", "intervention_feedback", "signals"} <= cols

    cid = database.save_concept("沉没成本")
    database.update_concept(cid, signals="{}")
    assert database.get_concept(cid)["signals"] == "{}"
