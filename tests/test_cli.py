"""Smoke tests for the CLI flow (scripted input + fake client, no network)."""

from __future__ import annotations

import json

import pytest

from cli import main as cli_main
from core import database
from core.client import DeepSeekError


class FakeClient:
    """Scripted DeepSeek client that answers from a list of question dicts."""

    def __init__(self, questions: list[dict], related=()) -> None:
        self.questions = list(questions)
        self.related = list(related)
        self.i = 0

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def chat(self, messages, **kwargs) -> str:
        user = messages[1]["content"]
        if "判断学习者的回答是否抓住了要点" in user:
            q = self.questions[self.i]
            self.i += 1
            return json.dumps(
                {"is_correct": q["correct"], "feedback": q["feedback"], "hint": q.get("hint")},
                ensure_ascii=False,
            )
        if "参考解释" in user:
            return "参考解释：机会成本是放弃的价值"
        if "concept_title" in user:
            return json.dumps(
                [{"concept_title": c, "relation_text": "都关于选择"} for c in self.related],
                ensure_ascii=False,
            )
        if "每日总结" in user:
            return json.dumps(
                {"breakthrough": "我终于搞懂了机会成本", "tomorrow_hook": "明天再想"},
                ensure_ascii=False,
            )
        if "层追问" in user:
            return f"问题：{self.questions[self.i]['question']}"
        raise AssertionError(f"unexpected prompt: {user[:40]}")


class BoomClient(FakeClient):
    def chat(self, messages, **kwargs) -> str:
        raise DeepSeekError("模拟 API 故障")


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    database.configure(tmp_path / "test.db")
    yield


def run_cli(monkeypatch, inputs: list[str], client_factory, argv: list[str] | None = None):
    answers = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(cli_main, "DeepSeekClient", client_factory)
    cli_main.main(argv)


def make_questions(n: int) -> list[dict]:
    return [{"question": f"问题{i}", "correct": True, "feedback": "对"} for i in range(n)]


def test_cli_full_flow(capsys, monkeypatch) -> None:
    database.save_concept("沉没成本", "已学")
    client_factory = lambda: FakeClient(
        make_questions(4), related=["沉没成本"]
    )
    run_cli(
        monkeypatch,
        ["机会成本", "原文：选择意味着放弃", "答1", "答2", "答3", "答4", "机会成本是放弃的价值"],
        client_factory,
        argv=[],
    )
    out = capsys.readouterr().out
    assert "我终于搞懂了机会成本" in out
    assert "沉没成本" in out  # connection recommendation
    assert "明天再想" in out  # tomorrow hook
    today = database.get_today_summary()
    assert today is not None
    assert today["breakthrough_text"] == "我终于搞懂了机会成本"
    target = next(c for c in database.get_all_concepts() if c["title"] == "机会成本")
    assert target["mastery"] == "搞懂了"


def test_cli_quit_mid_session(capsys, monkeypatch) -> None:
    run_cli(
        monkeypatch, ["机会成本", "原文", "q"], lambda: FakeClient(make_questions(4)), argv=[]
    )
    out = capsys.readouterr().out
    assert "AI 调用失败" not in out
    assert database.get_today_summary() is None


def test_cli_handles_api_error(capsys, monkeypatch) -> None:
    run_cli(
        monkeypatch,
        ["机会成本", "原文"],
        lambda: BoomClient(make_questions(4)),
        argv=[],
    )
    out = capsys.readouterr().out
    assert "AI 调用失败" in out
    assert "模拟 API 故障" in out


def test_read_line_quit_word(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "退出")
    assert cli_main.read_line(">") is None
    monkeypatch.setattr("builtins.input", lambda prompt="": "机会成本")
    assert cli_main.read_line(">") == "机会成本"


def test_history_lists_concepts(monkeypatch, capsys) -> None:
    database.save_concept("机会成本", "原文")
    cid = database.save_concept("沉没成本", "原文")
    database.update_concept(cid, mastery="搞懂了")
    inputs = iter(["q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    cli_main.main(["--history"])
    out = capsys.readouterr().out
    assert "历史回顾" in out
    assert "机会成本" in out
    assert "沉没成本" in out


def test_history_empty(monkeypatch, capsys) -> None:
    inputs = iter([])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    cli_main.main(["--history"])
    assert "还没有学习记录" in capsys.readouterr().out


def test_history_show_detail(monkeypatch, capsys) -> None:
    cid = database.save_concept("机会成本", "原文：选择意味着放弃")
    database.update_concept(cid, mastery="搞懂了", user_definition="机会成本是放弃的价值")
    database.save_qa(cid, "为什么机会成本和选择有关？", "因为每次都要放弃", True)
    database.save_qa(cid, "它关注过去还是未来？", "未来", False, hint_used=True)
    other = database.save_concept("沉没成本")
    database.save_connection(cid, other, "都关于选择，一个看未来一个看过去")
    database.save_daily_summary(cid, "我终于搞懂了机会成本", "明天再想")

    inputs = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    cli_main.main(["--history"])
    out = capsys.readouterr().out
    assert "我的理解：机会成本是放弃的价值" in out
    assert "为什么机会成本和选择有关？" in out
    assert "（用过提示）" in out
    assert "沉没成本" in out  # connection
    assert "我终于搞懂了机会成本" in out  # summary
    assert "明天再想" in out


def test_history_invalid_number(monkeypatch, capsys) -> None:
    database.save_concept("机会成本")
    inputs = iter(["99", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    cli_main.main(["--history"])
    assert "无效输入" in capsys.readouterr().out
