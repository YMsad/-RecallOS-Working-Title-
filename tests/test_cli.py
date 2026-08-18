"""Smoke tests for the CLI flow (scripted input + fake client, no network)."""

from __future__ import annotations

import json

import pytest

from cli import main as cli_main
from core import database
from core.client import DeepSeekAuthError, DeepSeekError
from core.config import get_api_key_from_config, reset_settings_cache


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
        if "hit the point" in user:
            q = self.questions[self.i]
            self.i += 1
            return json.dumps(
                {"is_correct": q["correct"], "feedback": q["feedback"], "hint": q.get("hint")},
                ensure_ascii=False,
            )
        if "reference explanation" in user:
            return "Reference explanation: opportunity cost is the value you give up"
        if "concept_title" in user:
            return json.dumps(
                [{"concept_title": c, "relation_text": "both about choice"} for c in self.related],
                ensure_ascii=False,
            )
        if "daily summary" in user:
            return json.dumps(
                {"breakthrough": "I finally understood opportunity cost",
                 "tomorrow_hook": "More to think about tomorrow"},
                ensure_ascii=False,
            )
        if "I don't get it" in user:
            return "In plain words: opportunity cost is the next-best choice you gave up"
        if "warm-up" in user:
            return "In one sentence, opportunity cost is the B you gave up to get A."
        if "Simplify this question" in user:
            return "Simplified question"
        if "different angle" in user:
            return "A question from a different angle"
        if "very first question" in user:
            return f"Opening question: {self.questions[self.i]['question']}"
        if "Layer" in user:
            return f"Question: {self.questions[self.i]['question']}"
        raise AssertionError(f"unexpected prompt: {user[:40]}")


class BoomClient(FakeClient):
    def chat(self, messages, **kwargs) -> str:
        raise DeepSeekError("simulated API failure")


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
    return [
        {"question": f"Question {i}", "correct": True, "feedback": "Right"} for i in range(n)
    ]


def test_cli_full_flow(capsys, monkeypatch) -> None:
    database.save_concept("sunk cost", "already learned")
    client_factory = lambda: FakeClient(
        make_questions(4), related=["sunk cost"]
    )
    run_cli(
        monkeypatch,
        [
            "opportunity cost",
            "Source: choice means giving up",
            "Answer 1",
            "Answer 2",
            "Answer 3",
            "Answer 4",
            "opportunity cost is the value of what you give up",
        ],
        client_factory,
        argv=[],
    )
    out = capsys.readouterr().out
    assert "I finally understood opportunity cost" in out
    assert "sunk cost" in out  # connection recommendation
    assert "More to think about tomorrow" in out  # tomorrow hook
    today = database.get_today_summary()
    assert today is not None
    assert today["breakthrough_text"] == "I finally understood opportunity cost"
    target = next(c for c in database.get_all_concepts() if c["title"] == "opportunity cost")
    assert target["mastery"] == "Understood"


def test_cli_quit_mid_session(capsys, monkeypatch) -> None:
    run_cli(
        monkeypatch,
        ["opportunity cost", "Source", "q"],
        lambda: FakeClient(make_questions(4)),
        argv=[],
    )
    out = capsys.readouterr().out
    assert "AI call failed" not in out
    assert database.get_today_summary() is None


def test_cli_handles_api_error(capsys, monkeypatch) -> None:
    run_cli(
        monkeypatch,
        ["opportunity cost", "Source"],
        lambda: BoomClient(make_questions(4)),
        argv=[],
    )
    out = capsys.readouterr().out
    assert "AI call failed" in out
    assert "simulated API failure" in out


def test_read_line_quit_word(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "退出")
    assert cli_main.read_line(">") is None
    monkeypatch.setattr("builtins.input", lambda prompt="": "opportunity cost")
    assert cli_main.read_line(">") == "opportunity cost"


def test_history_lists_concepts(monkeypatch, capsys) -> None:
    database.save_concept("opportunity cost", "source")
    cid = database.save_concept("sunk cost", "source")
    database.update_concept(cid, mastery="Understood")
    inputs = iter(["q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    cli_main.main(["--history"])
    out = capsys.readouterr().out
    assert "RecallOS History" in out
    assert "opportunity cost" in out
    assert "sunk cost" in out


def test_history_empty(monkeypatch, capsys) -> None:
    inputs = iter([])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    cli_main.main(["--history"])
    assert "No learning records yet" in capsys.readouterr().out


def test_history_show_detail(monkeypatch, capsys) -> None:
    cid = database.save_concept("opportunity cost", "Source: choice means giving up")
    database.update_concept(
        cid,
        mastery="Understood",
        user_definition="opportunity cost is the value of what you give up",
    )
    database.save_qa(
        cid, "Why is opportunity cost related to choice?", "because every choice means giving up", True
    )
    database.save_qa(cid, "Does it look to the past or the future?", "the future", False, hint_used=True)
    other = database.save_concept("sunk cost")
    database.save_connection(
        cid, other, "both about choice, one looks to the future and the other to the past"
    )
    database.save_daily_summary(
        cid, "I finally understood opportunity cost", "More to think about tomorrow"
    )

    inputs = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    cli_main.main(["--history"])
    out = capsys.readouterr().out
    assert "My understanding: opportunity cost is the value of what you give up" in out
    assert "Why is opportunity cost related to choice?" in out
    assert "(used hint)" in out
    assert "sunk cost" in out  # connection
    assert "I finally understood opportunity cost" in out  # summary
    assert "More to think about tomorrow" in out


def test_history_invalid_number(monkeypatch, capsys) -> None:
    database.save_concept("opportunity cost")
    inputs = iter(["99", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    cli_main.main(["--history"])
    assert "Invalid input" in capsys.readouterr().out


class PingClient:
    """Minimal client that answers the API-key validation ping."""

    def __enter__(self) -> "PingClient":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def chat(self, messages, **kwargs) -> str:
        return "OK"


def no_key_env(monkeypatch, tmp_path):
    """Simulate a machine with no key: empty .env + empty config file."""
    from core.config import Settings

    monkeypatch.setattr("core.config.CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("core.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("core.config.Settings", lambda: Settings(_env_file=None))
    reset_settings_cache()


def test_cli_prompts_for_key_when_missing(monkeypatch, capsys, tmp_path) -> None:
    no_key_env(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_main, "DeepSeekClient", lambda: PingClient())
    inputs = iter(["sk-cli-saved", "q"])
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(inputs)

    monkeypatch.setattr("builtins.input", fake_input)
    cli_main.main([])
    assert any("DeepSeek API Key" in p for p in prompts)
    assert get_api_key_from_config() == "sk-cli-saved"


def test_cli_reprompts_on_invalid_key(monkeypatch, capsys, tmp_path) -> None:
    no_key_env(monkeypatch, tmp_path)
    state = {"failed": False}

    def flaky_client():
        class FlakyClient(PingClient):
            def chat(self, messages, **kwargs) -> str:
                if not state["failed"]:
                    state["failed"] = True
                    raise DeepSeekAuthError("Authentication failed (HTTP 401)")
                return "OK"

        return FlakyClient()

    monkeypatch.setattr(cli_main, "DeepSeekClient", flaky_client)
    inputs = iter(["sk-bad", "sk-good", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    cli_main.main([])
    out = capsys.readouterr().out
    assert "Invalid key, please try again" in out
    assert get_api_key_from_config() == "sk-good"
