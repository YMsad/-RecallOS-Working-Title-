"""Tests for the Streamlit UI using streamlit.testing.v1.AppTest."""

from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

from core import database
from core.client import DeepSeekNetworkError

APP = "app.py"


class FakeClient:
    """Scripted DeepSeek client reused for the UI flow."""

    def __init__(self, questions: list[dict], related=()) -> None:
        self.questions = list(questions)
        self.related = list(related)
        self.i = 0
        self.depth = 0
        # V0.3.0 Learning Loop v2 fixed replies
        self.task_reply = {
            "task": "Explain opportunity cost to a friend in one sentence",
            "type": "summary",
            "difficulty": 2,
        }
        self.analysis_reply = {
            "understanding_level": "relationship",
            "understood": ["understands the core meaning"],
            "uncertain": [],
            "misconceptions": [],
            "last_response_quality": "deep",
        }
        self.decider_reply = {
            "action": "hint",
            "reason": "",
            "content": 'What does the "cost" in opportunity cost refer to?',
            "requires_user_response": True,
        }
        self.update_reply = {
            "understanding_level": "relationship",
            "understood": ["understands the core meaning"],
            "uncertain": [],
            "misconceptions": [],
            "last_response_quality": "deep",
            "next_best_action": "none",
        }
        self.offer_reply = {
            "offer": "Want to go one layer deeper?",
            "options": ["Go deeper", "That's enough"],
        }

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
            return "Reference: opportunity cost is the value you gave up"
        if "concept_title" in user:
            return json.dumps(
                [{"concept_title": c, "relation_text": "Both are about choices"} for c in self.related],
                ensure_ascii=False,
            )
        if "daily summary" in user:
            return json.dumps(
                {"breakthrough": "I finally got opportunity cost", "tomorrow_hook": "Think again tomorrow"},
                ensure_ascii=False,
            )
        if "review day" in user:
            return "Review question: what is opportunity cost?"
        if "I don't get it" in user:
            return "Plain words: opportunity cost is the next-best choice you gave up"
        if "warm-up" in user:
            return "In one sentence: opportunity cost is the B you give up to get A."
        if "Simplify this question" in user:
            return "Simplified question"
        if "different angle" in user:
            return "Question from a different angle"
        if "very first question" in user:
            return f"Opening question {self.i + 1}"
        if "Layer" in user:
            return f"Problem {self.i + 1}"
        if "not simple enough" in user:
            return "Simpler plain words: opportunity cost is the B you give up for A."
        if "Deeper probe" in user:
            self.depth += 1
            return f"Deeper problem {self.depth}"
        # ---- V0.3.0 Learning Loop v2 — prompt branches ----
        if "learning-task designer" in user:
            return json.dumps(self.task_reply, ensure_ascii=False)
        if "learner-state analyzer" in user:
            return json.dumps(self.analysis_reply, ensure_ascii=False)
        if "Socratic mentor" in user:
            return (
                "**Your answer mentioned:**\n- decisions\n\n"
                "**But the core of opportunity cost is:**\nthe best alternative you gave up\n\n"
                "**Try this example:**\nbuying a movie ticket\n\n"
                "**So:**\ndirection is right but needs adjusting"
            )
        if "minimal-intervention decider" in user:
            return json.dumps(self.decider_reply, ensure_ascii=False)
        if "learner-state updater" in user:
            return json.dumps(self.update_reply, ensure_ascii=False)
        if "learning coach" in user:
            return json.dumps(self.offer_reply, ensure_ascii=False)
        raise AssertionError(f"unexpected prompt: {user[:40]}")


class FlakyClient(FakeClient):
    """FakeClient that can simulate AI timeout/network failures on demand."""

    def __init__(self, questions, *, fail_task: int = 0, fail_analyze: int = 0,
                 fail_decide: int = 0, fail_offer: int = 0) -> None:
        super().__init__(questions)
        self.fail_task = fail_task
        self.fail_analyze = fail_analyze
        self.fail_decide = fail_decide
        self.fail_offer = fail_offer
        self._task_calls = 0
        self._analyze_calls = 0
        self._decide_calls = 0
        self._offer_calls = 0

    def chat(self, messages, **kwargs) -> str:
        user = messages[1]["content"]
        if "learning-task designer" in user:
            self._task_calls += 1
            if self._task_calls <= self.fail_task:
                raise DeepSeekNetworkError("Request timed out: simulated timeout")
        if "learner-state analyzer" in user:
            self._analyze_calls += 1
            if self._analyze_calls <= self.fail_analyze:
                raise DeepSeekNetworkError("Request timed out: simulated timeout")
        if "minimal-intervention decider" in user:
            self._decide_calls += 1
            if self._decide_calls <= self.fail_decide:
                raise DeepSeekNetworkError("Request timed out: simulated timeout")
        if "learning coach" in user:
            self._offer_calls += 1
            if self._offer_calls <= self.fail_offer:
                raise DeepSeekNetworkError("Request timed out: simulated timeout")
        return super().chat(messages, **kwargs)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    database.configure(tmp_path / "test.db")
    yield


@pytest.fixture
def configured_app(monkeypatch, tmp_path):
    """Give the app a valid API key so it boots to home (not the setup page)."""
    from core.config import Settings

    monkeypatch.setattr("core.config.Settings", lambda: Settings(deepseek_api_key="test-key", _env_file=None))
    from core import config

    config.reset_settings_cache()
    yield


@pytest.fixture
def legacy_flow(monkeypatch):
    """V0.3.0 — pin the legacy flow (RECALLOS_NEW_FLOW=0) so the old four-layer
    tests keep exercising it. AppTest runs app.py in a fresh process, so the
    switch must go through an environment variable, not a module attribute."""
    monkeypatch.setenv("RECALLOS_NEW_FLOW", "0")
    yield


def make_questions(n: int) -> list[dict]:
    return [{"question": f"Q{i}", "correct": True, "feedback": "Right"} for i in range(n)]


def current_step(at: AppTest) -> str:
    return at.session_state.step if "step" in at.session_state else "home"


def click_by_label(at: AppTest, text: str) -> AppTest:
    """Click the button whose label contains ``text`` and re-run."""
    for button in list(at.button):
        if text in (button.label or ""):
            return button.click().run()
    raise AssertionError(f"no button with label containing {text!r}")


def markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_app_boots_to_home(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert current_step(at) == "home"
    assert "What do you want to understand today?" in markdown_text(at)


def test_app_shows_key_setup_when_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("core.config.CONFIG_FILE", tmp_path / "cfg.json")
    monkeypatch.setattr("core.config.CONFIG_DIR", tmp_path)
    from core.config import Settings

    monkeypatch.setattr("core.config.Settings", lambda: Settings(_env_file=None))
    from core import config

    config.reset_settings_cache()
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert "First time here? Configure your DeepSeek API Key" in markdown_text(at)
    assert len(at.text_input) == 1
    assert at.text_input[0].label == "DeepSeek API Key"


def test_app_save_key_returns_to_home(monkeypatch, tmp_path) -> None:
    cfg = tmp_path / "cfg.json"
    monkeypatch.setattr("core.config.CONFIG_FILE", cfg)
    monkeypatch.setattr("core.config.CONFIG_DIR", tmp_path)
    from core.config import Settings

    monkeypatch.setattr("core.config.Settings", lambda: Settings(_env_file=None))
    from core import config

    config.reset_settings_cache()
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("sk-app-saved")
    at = click_by_label(at, "Save")
    assert not at.exception
    assert "First time here? Configure your DeepSeek API Key" not in markdown_text(at)
    assert config.get_api_key_from_config() == "sk-app-saved"


def test_app_full_flow(monkeypatch, configured_app, legacy_flow) -> None:
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient(make_questions(4)))
    at = AppTest.from_file(APP, default_timeout=30).run()

    # home -> start a session
    at.text_input[0].input("Opportunity Cost")
    at.text_area[0].input("Source: choice means giving up")
    at = click_by_label(at, "Start")
    assert not at.exception
    assert current_step(at) == "learning"

    # answer the 4 layer questions
    for _ in range(4):
        at = at.chat_input[0].set_value("Ans").run()
        assert not at.exception
    assert current_step(at) == "connections"

    # connections -> summary
    at = click_by_label(at, "Go to summary")
    assert current_step(at) == "summary"

    # summary: give own words, generate
    at.text_input[0].input("Opportunity cost is the value given up")
    at = click_by_label(at, "Generate summary")
    assert not at.exception
    assert "summary_result" in at.session_state

    today = database.get_today_summary()
    assert today is not None
    assert today["breakthrough_text"] == "I finally got opportunity cost"
    assert database.get_setting("streak") == "1"

    # summary view shows tomorrow hook
    assert "Think again tomorrow" in markdown_text(at)


def test_app_history_view(monkeypatch, configured_app) -> None:
    cid = database.save_concept("Opportunity Cost", "Source")
    database.update_concept(cid, mastery="Understood")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    assert current_step(at) == "history"
    assert "Opportunity Cost" in markdown_text(at)
    # the table shows mastery labels per row plus the column headers
    assert "✅ Understood" in markdown_text(at)
    assert "Concept" in markdown_text(at)
    assert "Actions" in markdown_text(at)


def test_app_mode_toggle_on_home(configured_app, legacy_flow) -> None:
    """V0.3.1 — the mode toggle is fully removed: even under the legacy flow the
    home page no longer shows a "has a foundation" switch."""
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert len(at.toggle) == 0


def test_app_learning_flow_uses_dynamic_opening(monkeypatch, configured_app, legacy_flow) -> None:
    fake = FakeClient(make_questions(4))
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("Opportunity Cost")
    at.text_area[0].input("Source: choice means giving up")
    at = click_by_label(at, "Start")
    assert not at.exception
    assert current_step(at) == "learning"
    assert "Opening question" in markdown_text(at)


def test_app_explain_button(monkeypatch, configured_app, legacy_flow) -> None:
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient(make_questions(4)))
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("Opportunity Cost")
    at.text_area[0].input("Source: choice means giving up")
    at = click_by_label(at, "Start")
    at = click_by_label(at, "I don't get it")
    assert not at.exception
    assert "Let me put it differently" in markdown_text(at)


def test_app_single_assistant_bubble_per_answer(monkeypatch, configured_app, legacy_flow) -> None:
    """One answer produces exactly one AI bubble (feedback + next question merged),
    never two bubbles."""
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient(make_questions(4)))
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("Opportunity Cost")
    at.text_area[0].input("Source: choice means giving up")
    at = click_by_label(at, "Start")
    at = at.chat_input[0].set_value("Ans").run()
    assert not at.exception
    msgs = at.session_state["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["assistant", "user", "assistant"]
    assert "✓ Right" in msgs[2]["text"]
    assert "Problem 2" in msgs[2]["text"]


def test_app_warmup_button_shows_prewarm_text(monkeypatch, configured_app) -> None:
    fake = FakeClient(make_questions(4))
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("Opportunity Cost")
    at.text_area[0].input("Source: choice means giving up")
    at = at.run()
    at = click_by_label(at, "Warm up")
    assert not at.exception
    assert any(
        "In one sentence: opportunity cost is the B you give up to get A." in m.value
        for m in at.info
    )


def test_app_warmup_prepended_to_first_message(monkeypatch, configured_app, legacy_flow) -> None:
    fake = FakeClient(make_questions(4))
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("Opportunity Cost")
    at.text_area[0].input("Source: choice means giving up")
    at = at.run()
    at = click_by_label(at, "Warm up")
    at = click_by_label(at, "Start")
    assert not at.exception
    assert current_step(at) == "learning"
    assert "In one sentence" in markdown_text(at)
    assert "Opening question" in markdown_text(at)


def test_app_reconfigure_dialog_opens(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    at = click_by_label(at, "Reconfigure API Key")
    assert not at.exception
    assert current_step(at) == "reconfigure"
    assert any(t.label == "New DeepSeek API Key" for t in at.text_input)
    assert any("Save and reload" in (b.label or "") for b in at.button)


def test_app_reconfigure_saves_new_key_and_returns_home(configured_app) -> None:
    from core import config

    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "Reconfigure API Key")
    key_input = next(t for t in at.text_input if t.label == "New DeepSeek API Key")
    key_input.input("sk-new-key-123")
    at = at.run()
    at = click_by_label(at, "Save and reload")
    assert not at.exception
    assert config.get_api_key_from_config() == "sk-new-key-123"
    assert current_step(at) == "home"


# ------------------------------------------------------------------ V0.2.2


def _seed_due_concept() -> int:
    """Create a concept that is due for review today."""
    from datetime import date

    cid = database.save_concept("Opportunity Cost", "Source: choice means giving up")
    database.update_concept(cid, mastery="Understood", next_review_date=date.today().isoformat())
    return cid


def test_app_review_entry_shown_on_home(configured_app) -> None:
    _seed_due_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert any("Review today" in (b.label or "") for b in at.button)


def test_app_review_entry_hidden_when_nothing_due(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert not any("Review today" in (b.label or "") for b in at.button)


def test_app_review_list_shows_due_concepts(configured_app) -> None:
    _seed_due_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "Review today")
    assert not at.exception
    assert current_step(at) == "review_list"
    assert "Opportunity Cost" in markdown_text(at)
    assert any("Start review" in (b.label or "") for b in at.button)


def test_app_review_mode_asks_tomorrow_hook_and_passes(monkeypatch, configured_app) -> None:
    from core import review as review_module

    cid = _seed_due_concept()
    database.save_daily_summary(cid, "I finally got it", "Think again tomorrow")
    fake = FakeClient([{"question": "Q", "correct": True, "feedback": "Right"}] * 3)
    monkeypatch.setattr(review_module, "DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "Review today")
    at = click_by_label(at, "Start review")
    assert not at.exception
    assert current_step(at) == "review"
    assert "Think again tomorrow" in markdown_text(at)

    at = at.chat_input[0].set_value("Ans").run()
    assert not at.exception
    assert "Right" in markdown_text(at)
    assert any("Review passed" in (s.value or "") for s in at.success)


def test_app_review_mode_three_failures_needs_relearn(monkeypatch, configured_app) -> None:
    from core import review as review_module

    cid = _seed_due_concept()
    database.save_daily_summary(cid, "I finally got it", "Think again tomorrow")
    fake = FakeClient([{"question": "Q", "correct": False, "feedback": "Try again"}] * 3)
    monkeypatch.setattr(review_module, "DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "Review today")
    at = click_by_label(at, "Start review")
    assert not at.exception
    for _ in range(3):
        at = at.chat_input[0].set_value("Wrong").run()
    assert not at.exception
    assert "re-learned" in markdown_text(at)
    assert any("re-learning" in (w.value or "") for w in at.warning)
    assert database.get_concept(cid)["mastery"] == "Learning"


def test_app_connection_bidirectional_jump(monkeypatch, configured_app) -> None:
    cid_a = database.save_concept("Opportunity Cost", "Source A")
    cid_b = database.save_concept("Sunk Cost", "Source B")
    database.save_connection(cid_a, cid_b, "Both are about choices")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    assert current_step(at) == "history"
    assert any("Go to" in (b.label or "") for b in at.button)
    at = click_by_label(at, "Go to")
    assert not at.exception
    assert current_step(at) == "concept_detail"
    assert "Sunk Cost" in markdown_text(at)


def test_app_connection_long_text_preserved(monkeypatch, configured_app) -> None:
    cid_a = database.save_concept("Opportunity Cost", "Source A")
    cid_b = database.save_concept("Sunk Cost", "Source B")
    database.save_connection(cid_a, cid_b, "Both are about choices")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    # the full relationship description should be shown, not truncated to one line
    assert "Both are about choices" in markdown_text(at)


# --------------------------------------------------------------- usage stats


def test_app_usage_stats_page(configured_app) -> None:
    database.save_usage_log(model="deepseek-chat", prompt_tokens=100,
                            completion_tokens=50, total_tokens=150, cost=0.001)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "Usage")
    assert not at.exception
    assert current_step(at) == "usage_stats"
    assert "Usage statistics" in markdown_text(at)
    # cumulative / month / today should each show a Token number (metric value)
    metric_values = [m.value for m in at.metric]
    assert len(metric_values) == 3
    assert any("150" in (v or "") for v in metric_values)


def test_app_usage_stats_empty(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "Usage")
    assert not at.exception
    assert current_step(at) == "usage_stats"
    assert any("No usage data yet" in (i.value or "") for i in at.info)


# ---------------------------------------------------------------- V0.2.3


def _seed_unfinished_concept() -> int:
    cid = database.save_concept("Opportunity Cost", "Source: choice means giving up")
    database.save_qa(cid, "Does it look at the past or the future?", "The future", True)
    database.save_qa(cid, "How is it different from sunk cost?", "It looks at the past", False, hint_used=True)
    return cid


def test_continue_learning_entry(configured_app) -> None:
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not any("Keep learning" in (b.label or "") for b in at.button)

    _seed_unfinished_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert any("Keep learning: Opportunity Cost" in (b.label or "") for b in at.button)


def test_continue_learning_restores_messages(configured_app) -> None:
    cid = _seed_unfinished_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "Keep learning")
    assert not at.exception
    assert current_step(at) == "learning"

    session = at.session_state["session"]
    assert session.concept_id == cid

    msgs = at.session_state["messages"]
    assert len(msgs) == 4
    assert msgs[0] == {"role": "assistant", "text": "Does it look at the past or the future?"}
    assert msgs[1] == {"role": "user", "text": "The future"}
    assert msgs[2] == {"role": "assistant", "text": "How is it different from sunk cost?"}
    assert msgs[3] == {"role": "user", "text": "It looks at the past"}
    assert "Does it look at the past or the future?" in markdown_text(at)


def test_app_delete_concept_flow(configured_app) -> None:
    cid = database.save_concept("Opportunity Cost", "Source")
    database.save_qa(cid, "Q?", "A", True)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    at = click_by_label(at, "✕")
    assert not at.exception
    assert any("Delete" in (b.label or "") for b in at.button)

    at = click_by_label(at, "Delete")
    assert not at.exception
    assert current_step(at) == "history"
    assert database.get_concept(cid) is None
    assert database.get_qa_history(cid) == []
    assert any("No learning records yet" in (i.value or "") for i in at.info)


def test_app_delete_from_table(configured_app) -> None:
    a = database.save_concept("Opportunity Cost", "Source A")
    b = database.save_concept("Sunk Cost", "Source B")
    database.update_concept(a, mastery="Understood")
    database.save_qa(b, "Q?", "A", True)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    assert current_step(at) == "history"
    # the table shows mastery labels per row (Understood / Learning), no "Unclear" row
    assert "✅ Understood" in markdown_text(at)
    assert "📖 Learning" in markdown_text(at)
    assert "🔄 Unclear" not in markdown_text(at)
    # the row's "View" button toggles the inline detail
    at = at.button(key=f"view_{b}").click().run()
    assert not at.exception
    assert "history_view_id" in at.session_state
    assert at.session_state["history_view_id"] == b
    assert "Sunk Cost" in markdown_text(at)
    # the row's "✕" starts the two-step delete of concept A; the list refreshes
    at = at.button(key=f"xdel_{a}").click().run()
    assert not at.exception
    assert any("Delete" in (btn.label or "") for btn in at.button)
    at = click_by_label(at, "Delete")
    assert not at.exception
    assert current_step(at) == "history"
    assert database.get_concept(a) is None
    assert database.get_concept(b) is not None
    assert "Opportunity Cost" not in markdown_text(at)
    assert "📖 Learning" in markdown_text(at)


def test_app_history_view_and_delete_buttons_via_callback(configured_app) -> None:
    """View / ✕ / confirm-delete all go through on_click callbacks: with several
    concepts in one group, switching and deleting take effect immediately."""
    a = database.save_concept("Opportunity Cost", "Source A")
    b = database.save_concept("Sunk Cost", "Source B")
    database.update_concept(a, mastery="Understood")
    database.update_concept(b, mastery="Understood")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    assert current_step(at) == "history"
    # clicking "View" switches the inline detail to b
    at = at.button(key=f"view_{b}").click().run()
    assert not at.exception
    assert at.session_state["history_view_id"] == b
    # clicking "✕" opens the confirm dialog, but nothing is deleted yet
    at = at.button(key=f"xdel_{a}").click().run()
    assert not at.exception
    assert any("Delete" in (btn.label or "") for btn in at.button)
    assert database.get_concept(a) is not None
    # clicking the confirm "Delete" deletes via the callback and refreshes the list
    at = click_by_label(at, "Delete")
    assert not at.exception
    assert database.get_concept(a) is None
    assert database.get_concept(b) is not None
    assert "Opportunity Cost" not in markdown_text(at)
    assert "Sunk Cost" in markdown_text(at)


def test_app_delete_from_detail_page_returns_to_history(configured_app) -> None:
    """Detail-page helper delete: after the two-step confirmation we're back on
    the history page and the deleted concept is gone."""
    cid_a = database.save_concept("Opportunity Cost", "Source A")
    cid_b = database.save_concept("Sunk Cost", "Source B")
    database.save_connection(cid_a, cid_b, "Both are about choices")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    at = click_by_label(at, "Go to")
    assert not at.exception
    assert current_step(at) == "concept_detail"
    shown = at.session_state["concept_detail_id"]
    # the detail page has a delete entry at the top, with two-step confirmation
    assert any("Delete this concept" in (b.label or "") for b in at.button)
    at = click_by_label(at, "Delete this concept")
    assert not at.exception
    assert any(b.key == f"confirm_ok_{shown}" for b in at.button)
    at = at.button(key=f"confirm_ok_{shown}").click().run()
    assert not at.exception
    assert current_step(at) == "history"
    assert database.get_concept(shown) is None


def test_app_history_view_expands_inline_and_collapses(configured_app) -> None:
    """"View" expands the detail inline right below the row; only one expands at a
    time; clicking again collapses it."""
    a = database.save_concept("Opportunity Cost", "Source A")
    b = database.save_concept("Sunk Cost", "Source B")
    database.update_concept(a, mastery="Understood")
    database.update_concept(b, mastery="Understood")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    assert current_step(at) == "history"
    # on first entry the first concept expands by default; only one area is open
    assert any("Close" in (btn.label or "") for btn in at.button)
    # clicking "View" switches the expansion to b (only one at a time)
    at = at.button(key=f"view_{b}").click().run()
    assert not at.exception
    assert at.session_state["history_view_id"] == b
    assert any("Close" in (btn.label or "") for btn in at.button)
    # clicking "View" on the currently-expanded concept collapses it
    at = at.button(key=f"view_{b}").click().run()
    assert not at.exception
    assert at.session_state["history_view_id"] is None
    assert not any("Close" in (btn.label or "") for btn in at.button)
    # we can expand another concept again
    at = at.button(key=f"view_{a}").click().run()
    assert not at.exception
    assert at.session_state["history_view_id"] == a


# ---------------------------------------------------------------- V0.3.0 UI


def _start_new_flow(at: AppTest) -> AppTest:
    at.text_input[0].input("Opportunity Cost")
    at.text_area[0].input("Source: choice means giving up")
    return click_by_label(at, "Start")


def test_app_new_flow_home_drops_mode_toggle(configured_app) -> None:
    """The new-flow home page no longer shows the "has a foundation" mode toggle."""
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert len(at.toggle) == 0
    assert any("Concept name" in (ti.label or "") for ti in at.text_input)


def test_app_new_flow_reading_to_validation(monkeypatch, configured_app) -> None:
    fake = FakeClient([{"question": "Q", "correct": True, "feedback": "Passed"}])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    assert not at.exception
    assert current_step(at) == "learning"
    session = at.session_state["session"]
    assert session.flow == "new"
    assert session.stage == "reading"
    assert any("Reading" in (i.value or "") for i in at.info)
    assert any("First read the source" in m["text"] for m in at.session_state["messages"])

    at = click_by_label(at, "I've finished reading")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"
    assert any("Check your understanding" in m["text"] for m in at.session_state["messages"])
    assert any("I can't understand this" in (b.label or "") for b in at.button)

    # "I can't understand this" -> plain-language explanation, stage unchanged
    at = click_by_label(at, "I can't understand this")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"
    assert any("Simpler plain words" in m["text"] for m in at.session_state["messages"])


def test_app_new_flow_home_goal_selector(configured_app) -> None:
    """V0.3.1 — the home page offers a learning goal; default is "understand a
    concept"; picking one persists it."""
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert at.radio[0].value == "🧠 Understand a concept"

    at.radio[0].set_value("🛠 Apply in practice").run()
    assert not at.exception
    at = _start_new_flow(at)
    assert not at.exception
    session = at.session_state["session"]
    assert session.learning_goal == "apply"
    assert database.get_concept(session.concept_id)["learning_goal"] == "apply"


def test_app_new_flow_reading_split_with_guidance(monkeypatch, configured_app) -> None:
    """V0.3.1 fix — reading split into paragraphs + one guiding question per
    paragraph: not just reading, but summarizing each part in your own words."""
    fake = FakeClient([])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    assert at.session_state["session"].stage == "reading"

    # split paragraphs + a guiding text_input per paragraph
    assert "Paragraph 1 / 1" in markdown_text(at)
    assert any(t.key == "v_read_ans_0" for t in at.text_input)
    assert any("What does this paragraph say?" in (t.label or "") for t in at.text_input)
    # the old reading-signal buttons (🤔💡✓) and stuck-point input are gone
    assert not any(b.key and b.key.startswith("v_rs_") for b in at.button)
    assert not any(t.key == "v_stuck_text" for t in at.text_input)

    # reading doesn't block: validation can start normally
    at = click_by_label(at, "I've finished reading")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"


def test_app_new_flow_reading_answer_recorded(monkeypatch, configured_app) -> None:
    """V0.3.1 fix — the per-paragraph guiding answer is recorded and persisted
    into reading_answers."""
    fake = FakeClient([])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    assert at.session_state["session"].stage == "reading"

    at.text_input[0].input("Opportunity cost is the price of the choice you gave up").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.reading_answer_text(0) == "Opportunity cost is the price of the choice you gave up"
    saved = json.loads(database.get_concept(session.concept_id)["reading_answers"])
    assert saved[0]["paragraph_index"] == 0
    assert saved[0]["answer"] == "Opportunity cost is the price of the choice you gave up"
    assert any("1/1 paragraphs noted" in (c.value or "") for c in at.caption)

    # starting the check still advances the stage; recording doesn't block it
    at = click_by_label(at, "I've finished reading")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"


def test_app_new_flow_reading_no_source_skips_to_validation(monkeypatch, configured_app) -> None:
    """V0.3.1 — with no source text, the reading stage offers "start the check"
    directly."""
    fake = FakeClient([])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("Opportunity Cost")
    at = click_by_label(at, "Start")
    assert not at.exception
    assert at.session_state["session"].stage == "reading"
    assert any("Start the check now" in (b.label or "") for b in at.button)
    at = click_by_label(at, "Start the check now")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"


def test_app_new_flow_reading_pages_one_paragraph(monkeypatch, configured_app) -> None:
    """V0.3.1 hotfix — reading one paragraph at a time: progress + prev/next
    navigation + "I've finished reading", no longer a single wall of text."""
    fake = FakeClient([])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("Opportunity Cost")
    at.text_area[0].input("Source: choice means giving up.\n\nOpportunity cost is the best choice that was given up.")
    at = click_by_label(at, "Start")
    assert not at.exception
    assert at.session_state["session"].stage == "reading"

    # only the first paragraph + progress + nav buttons, no chat bubbles
    assert at.session_state["v_read_index"] == 0
    assert "Paragraph 1 / 2" in markdown_text(at)
    assert any(t.key == "v_read_ans_0" for t in at.text_input)
    assert len(at.chat_message) == 0
    assert any("Previous" in (b.label or "") for b in at.button)
    assert any("Next" in (b.label or "") for b in at.button)

    # note the first paragraph, then switch to the second
    at.text_input[0].input("First paragraph summary").run()
    assert not at.exception
    at = click_by_label(at, "Next")
    assert not at.exception
    assert at.session_state["v_read_index"] == 1
    assert "Paragraph 2 / 2" in markdown_text(at)
    assert any(t.key == "v_read_ans_1" for t in at.text_input)
    assert at.session_state["session"].reading_answer_text(0) == "First paragraph summary"

    # can go back, and each paragraph's answer is kept per paragraph
    at = click_by_label(at, "Previous")
    assert not at.exception
    assert at.session_state["v_read_index"] == 0
    at.text_input[0].input("First paragraph summary (revised)").run()
    assert at.session_state["session"].reading_answer_text(0) == "First paragraph summary (revised)"

    # reading done -> start the check
    at = click_by_label(at, "I've finished reading")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"


def test_app_new_flow_reading_show_keys_collapses_paragraph(monkeypatch, configured_app) -> None:
    """V0.3.1 hotfix — reading "Key points only": shows only the key sentences
    and tucks the full text into an expander."""
    fake = FakeClient([])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.text_input[0].input("Custom Concept")
    at.text_area[0].input(
        "First sentence covers the core. Second sentence gives the details. "
        "Third sentence has the **key** phrase. Fourth sentence adds an example. "
        "Fifth sentence wraps it all up."
    )
    at = click_by_label(at, "Start")
    assert not at.exception
    assert at.session_state["session"].stage == "reading"

    # by default the full paragraph is shown directly
    assert "Fifth sentence wraps it all up" in markdown_text(at)
    assert not any("Expand full text" in (e.label or "") for e in at.expander)

    # with "Key points only" on, only the key sentences render; the full text goes
    # into the expander
    show_keys = next(t for t in at.toggle if "Key points only" in (t.label or ""))
    at = show_keys.set_value(True).run()
    assert not at.exception
    md = markdown_text(at)
    assert "Third sentence has the **key** phrase." in md
    assert "- First sentence covers the core." in md
    assert any("Expand full text" in (e.label or "") for e in at.expander)


def test_app_builtin_reading_bold_and_pagination(configured_app) -> None:
    """V0.3.1 hotfix — builtin concepts reading: bold keywords are preserved and
    multi-paragraph navigation shows one paragraph at a time."""
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = at.button(key="builtin_opportunity_cost").click().run()
    assert not at.exception
    assert at.session_state["session"].stage == "reading"

    assert "**Opportunity cost**" in markdown_text(at)
    assert "Paragraph 1 / 4" in markdown_text(at)
    assert any("Key points only" in (t.label or "") for t in at.toggle)
    assert any("Next" in (b.label or "") for b in at.button)

    at = click_by_label(at, "Next")
    assert not at.exception
    assert at.session_state["v_read_index"] == 1
    assert "Paragraph 2 / 4" in markdown_text(at)


def test_app_new_flow_complete_hides_chat_bubbles(monkeypatch, configured_app) -> None:
    """V0.3.1 hotfix — the complete stage shows only a success message + short
    confirmation + "View today's summary"; chat bubbles are hidden."""
    fake = FakeClient([{"question": "Q", "correct": True, "feedback": "Passed"}])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    assert not at.exception
    at.chat_input[0].set_value("Opportunity cost is the next-best choice given up").run()
    assert not at.exception
    assert at.session_state["session"].stage == "complete"

    assert any("Your understanding is confirmed" in (s.value or "") for s in at.success)
    assert any("I'll remind you to come back tomorrow" in (c.value or "") for c in at.caption)
    assert len(at.chat_message) == 0
    assert any("View today's summary" in (b.label or "") for b in at.button)


def test_app_new_flow_intervention_feedback_ui(monkeypatch, configured_app) -> None:
    """V0.3.1 — after an intervention the 👍/🤔 feedback buttons are optional;
    the feedback is written into the learner state and persisted."""
    fake = FakeClient([])
    fake.analysis_reply = {
        "understanding_level": "relationship",
        "understood": [],
        "uncertain": ["boundary unclear"],
        "misconceptions": [],
        "last_response_quality": "partial",
    }
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    at = at.chat_input[0].set_value("Opportunity cost is the next-best choice given up").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "intervention"
    assert session.feedback_pending() is True

    # the feedback buttons only render on the next clean run (the answer-handling
    # run hasn't reached the intervention stage at the top yet)
    at = at.run()
    assert not at.exception
    assert any("Much clearer" in (b.label or "") for b in at.button)

    # submit the "👍 Much clearer" feedback
    at = click_by_label(at, "Much clearer")
    assert not at.exception
    session = at.session_state["session"]
    assert session.intervention_feedback_list == [
        {"action": "hint", "feedback": "clear"}
    ]
    assert (
        session.learner_state.intervention_history[-1]["feedback"] == "clear"
    )
    assert database.get_concept(session.concept_id)["intervention_feedback"]


def test_app_new_flow_validation_pass_goes_complete(monkeypatch, configured_app) -> None:
    """V0.3.1 — passing the check ends it: no understanding gap -> complete, no
    "go deeper?" prompt."""
    fake = FakeClient([])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    at = at.chat_input[0].set_value("Opportunity cost is the next-best choice given up").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.validation_passed is True
    assert session.stage == "complete"
    assert session.phase == "connections"
    # completion notification bubble (with the level) + the complete page
    msgs = at.session_state["messages"]
    assert any("Your understanding is solid" in m["text"] for m in msgs)
    assert any("Your understanding is confirmed" in (s.value or "") for s in at.success)
    # the old offer "go deeper?" choices are gone
    assert not any("Go deeper" in (b.label or "") for b in at.button)
    assert not any("That's enough" in (b.label or "") for b in at.button)


def test_app_new_flow_validation_gap_enters_intervention(monkeypatch, configured_app) -> None:
    fake = FakeClient([])
    fake.analysis_reply = {
        "understanding_level": "relationship",
        "understood": [],
        "uncertain": ["boundary unclear"],
        "misconceptions": ["confused opportunity cost with sunk cost"],
        "last_response_quality": "partial",
    }
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    at = at.chat_input[0].set_value("Opportunity cost is a choice").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.validation_passed is False
    assert session.stage == "intervention"
    assert any('What does the "cost" in opportunity cost refer to?' in m["text"] for m in at.session_state["messages"])
    assert "Minimal intervention" in markdown_text(at)


def test_app_new_flow_intervention_answer_stops_when_understand(monkeypatch, configured_app) -> None:
    fake = FakeClient([])
    fake.analysis_reply = {
        "understanding_level": "relationship",
        "understood": [],
        "uncertain": ["boundary unclear"],
        "misconceptions": ["confused"],
        "last_response_quality": "partial",
    }
    fake.update_reply = {
        "understanding_level": "application",
        "understood": ["understands the core and the boundary"],
        "uncertain": [],
        "misconceptions": [],
        "last_response_quality": "deep",
        "next_best_action": "none",
    }
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    at = at.chat_input[0].set_value("Opportunity cost is a choice").run()  # gap -> intervention
    assert not at.exception
    assert at.session_state["session"].stage == "intervention"
    # answering the intervention -> updater says it's understood -> no more
    # interventions -> complete
    at = at.chat_input[0].set_value("The cost is the best choice you gave up").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "complete"
    assert session.phase == "connections"
    assert any("Your understanding is confirmed" in (s.value or "") for s in at.success)
    assert len(session.deeper_history) == 1
    assert session.deeper_history[0]["next_best_action"] == "none"


def test_app_new_flow_complete_goes_to_summary(monkeypatch, configured_app) -> None:
    """V0.3.1 — complete means summarize: after passing the check, "View today's
    summary" auto-generates the summary, no input needed."""
    fake = FakeClient([])
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    at = at.chat_input[0].set_value("Opportunity cost is the next-best choice given up").run()  # pass -> complete
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "complete"

    at = click_by_label(at, "View today's summary")
    assert not at.exception
    assert current_step(at) == "summary"
    # auto-generated: no manual input, the summary appears right on the page
    assert "summary_result" in at.session_state
    assert "Think again tomorrow" in markdown_text(at)
    today = database.get_today_summary()
    assert today is not None
    assert today["breakthrough_text"] == "I finally got opportunity cost"
    assert database.get_setting("streak") == "1"
    # the latest validation answer becomes "My understanding"
    concept = database.get_concept(session.concept_id)
    assert concept["user_definition"] == "Opportunity cost is the next-best choice given up"
    assert concept["mastery"] == "Understood"


class PlainSummaryClient(FakeClient):
    """FakeClient whose summary also includes the plain one-liner."""

    def chat(self, messages, **kwargs) -> str:
        user = messages[1]["content"]
        if "daily summary" in user:
            return json.dumps(
                {
                    "breakthrough": "I finally got opportunity cost",
                    "plain": "Put simply: choosing A means you lose B.",
                    "tomorrow_hook": "Think again tomorrow",
                },
                ensure_ascii=False,
            )
        return super().chat(messages, **kwargs)


def test_app_summary_shows_three_part_summary(monkeypatch, configured_app) -> None:
    """V0.3.1 fix — the summary page shows breakthrough / plain (optional) /
    tomorrow_hook."""
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: PlainSummaryClient([]))
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    at = at.chat_input[0].set_value("Opportunity cost is the next-best choice given up").run()
    at = click_by_label(at, "View today's summary")
    assert not at.exception
    text = markdown_text(at)
    assert "What you understood" in text
    assert "In plain words, it means" in text
    assert "Put simply: choosing A means you lose B." in text
    assert "Tomorrow the AI will ask you" in text
    assert "Think again tomorrow" in text


def test_app_summary_auto_timeout_shows_retry(monkeypatch, configured_app) -> None:
    """V0.3.1 — auto-summary AI timeout: no crash, a retry button appears, and
    retrying recovers."""
    fake = FlakyClient([])

    class FailingSummaryClient(FakeClient):
        def __init__(self, inner: FakeClient) -> None:
            super().__init__([])
            self._inner = inner
            self._calls = 0

        def chat(self, messages, **kwargs) -> str:
            user = messages[1]["content"]
            if "daily summary" in user:
                self._calls += 1
                if self._calls == 1:
                    raise DeepSeekNetworkError("Request timed out: simulated timeout")
            return self._inner.chat(messages, **kwargs)

    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FailingSummaryClient(fake))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    at = at.chat_input[0].set_value("Opportunity cost is the next-best choice given up").run()
    at = click_by_label(at, "View today's summary")
    assert not at.exception
    assert "summary_error" in at.session_state
    assert any("Retry" in (b.label or "") for b in at.button)
    at = click_by_label(at, "Retry")
    assert not at.exception
    assert "summary_result" in at.session_state


def test_app_validation_timeout_retry_recovers(monkeypatch, configured_app) -> None:
    """The validation-analysis AI times out: no spinner-loop or crash, an ❌
    bubble appears, and re-submitting the same answer recovers."""
    fake = FlakyClient([], fail_analyze=1)
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    at = at.chat_input[0].set_value("Opportunity cost is the next-best choice given up").run()
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "validation"  # failure doesn't advance
    assert any("❌" in m["text"] for m in at.session_state["messages"] if m["text"])
    # re-submitting the same answer works -> analysis succeeds -> complete (no gap)
    at = at.chat_input[0].set_value("Opportunity cost is the next-best choice given up").run()
    assert not at.exception
    assert at.session_state["session"].stage == "complete"


class JunkTaskClient(FakeClient):
    """FakeClient that returns non-JSON for the validation-task prompt."""

    def chat(self, messages, **kwargs) -> str:
        user = messages[1]["content"]
        if "learning-task designer" in user:
            return "Sorry — I didn't output in the required format this time"
        return super().chat(messages, **kwargs)


def test_app_start_validation_timeout_retryable(monkeypatch, configured_app) -> None:
    """The "start the check" AI times out after reading: no spinner-loop or crash,
    stays on the reading stage, retry recovers."""
    fake = FlakyClient([], fail_task=1)
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: fake)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "reading"  # failure doesn't advance; retry stays possible
    assert any("❌" in m["text"] for m in at.session_state["messages"] if m["text"])
    assert any("retry" in m["text"] for m in at.session_state["messages"] if m["text"])
    # the button is still there -> the user can click it again
    assert any("I've finished reading" in (b.label or "") for b in at.button)

    # retry -> succeeds on the second try -> validation stage
    at = click_by_label(at, "I've finished reading")
    assert not at.exception
    assert at.session_state["session"].stage == "validation"
    assert any("Check your understanding" in m["text"] for m in at.session_state["messages"])


def test_app_start_validation_malformed_reply_friendly_error(
    monkeypatch, configured_app
) -> None:
    """The validation-task AI returns non-JSON: no uncaught exception, it turns
    into a friendly message and stays retryable."""
    monkeypatch.setattr(
        "core.session.DeepSeekClient", lambda: JunkTaskClient([])
    )
    at = AppTest.from_file(APP, default_timeout=30).run()
    at = _start_new_flow(at)
    at = click_by_label(at, "I've finished reading")
    assert not at.exception
    session = at.session_state["session"]
    assert session.stage == "reading"  # no crash, no advance, still retryable
    assert any("was not in the expected format" in m["text"] for m in at.session_state["messages"] if m["text"])
    assert any("I've finished reading" in (b.label or "") for b in at.button)


# ----------------------------------------------------- V0.3.0 session resume


def _seed_new_flow_concept(**updates) -> int:
    """Create a concept saved in the middle of the V0.3.0 new flow."""
    cid = database.save_concept("Opportunity Cost", "Source: choice means giving up")
    fields = {
        "stage": "validation",
        "validation_kind": "summary",
        "validation_difficulty": 2,
        "validation_task": "Explain opportunity cost to a friend in one sentence",
        "validation_target": "Name the best alternative you gave up",
        "validation_passed": False,
        "validation_attempts": 1,
        "validation_history": json.dumps(
            [
                {
                    "answer": "First attempt",
                    "understanding_level": "relationship",
                    "last_response_quality": "partial",
                    "understood": [],
                    "uncertain": ["boundary unclear"],
                    "misconceptions": [],
                }
            ],
            ensure_ascii=False,
        ),
    }
    fields.update(updates)
    database.update_concept(cid, **fields)
    return cid


def test_app_continue_new_flow_resumes_validation(configured_app) -> None:
    cid = _seed_new_flow_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    at = click_by_label(at, "Keep learning")
    assert not at.exception
    assert current_step(at) == "learning"

    session = at.session_state["session"]
    assert session.flow == "new"
    assert session.stage == "validation"
    assert session.validation_attempts == 1
    assert session.validation_task == "Explain opportunity cost to a friend in one sentence"
    assert session.learner_state.understanding_level == "relationship"

    # the rebuilt bubbles: check task + previous answer + analyzed bubble
    msgs = at.session_state["messages"]
    assert any("Explain opportunity cost to a friend in one sentence" in m["text"] for m in msgs)
    assert any(m["text"] == "First attempt" for m in msgs)
    assert any("(level: relationship)" in m["text"] for m in msgs)
    # the UI renders the check task directly (not only inside a bubble)
    assert any("Explain opportunity cost to a friend in one sentence" in m.value for m in at.markdown)


def test_app_continue_new_flow_resumes_offer(monkeypatch, configured_app) -> None:
    """V0.3.1 — a leftover offer session (old data) resumes straight to complete,
    no longer asking "go deeper?"."""
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient([]))
    cid = _seed_new_flow_concept(
        stage="offer",
        validation_passed=True,
        validation_attempts=0,
        validation_history=json.dumps(
            [{
                "answer": "Opportunity cost is the next-best choice I gave up",
                "understanding_level": "relationship",
                "last_response_quality": "deep",
                "understood": ["core"],
                "uncertain": [],
                "misconceptions": [],
            }],
            ensure_ascii=False,
        ),
    )
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    at = click_by_label(at, "Keep learning")
    assert not at.exception
    assert current_step(at) == "learning"

    session = at.session_state["session"]
    assert session.flow == "new"
    assert session.stage == "complete"  # V0.3.1 — a leftover offer session counts as complete
    assert session.validation_passed is True
    # the complete page appears, no more "go deeper?" question
    assert any("Your understanding is confirmed" in (s.value or "") for s in at.success)
    assert not any("Go one layer deeper" in (b.label or "") for b in at.button)


def test_app_continue_new_flow_resumes_intervention(monkeypatch, configured_app) -> None:
    """Interrupted mid-intervention: on resume the next intervention is decided
    and pushed onto the screen automatically."""
    monkeypatch.setattr("core.session.DeepSeekClient", lambda: FakeClient([]))
    cid = _seed_new_flow_concept(
        stage="intervention",
        validation_passed=False,
        validation_attempts=1,
        validation_history=json.dumps(
            [{
                "answer": "First attempt",
                "understanding_level": "relationship",
                "last_response_quality": "partial",
                "understood": [],
                "uncertain": ["boundary unclear"],
                "misconceptions": ["confused"],
            }],
            ensure_ascii=False,
        ),
        deeper_answers=json.dumps(
            [{
                "question": 'Think about what "cost" refers to here',
                "answer": "The cost is the best alternative you gave up",
                "action": "hint",
                "understanding_level": "relationship",
                "last_response_quality": "partial",
                "understood": [],
                "uncertain": ["boundary"],
                "misconceptions": [],
                "next_best_action": "hint",
            }],
            ensure_ascii=False,
        ),
        deeper_index=1,
    )
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    at = click_by_label(at, "Keep learning")
    assert not at.exception
    assert current_step(at) == "learning"

    session = at.session_state["session"]
    assert session.flow == "new"
    assert session.stage == "intervention"
    assert session.current_intervention() is not None
    # the answered intervention is restored in the messages, and the next one is
    # auto-generated by the decider
    msgs = at.session_state["messages"]
    assert any('Think about what "cost" refers to here' in m["text"] for m in msgs)
    assert any('What does the "cost" in opportunity cost refer to?' in m["text"] for m in msgs)
    assert any("Minimal intervention" in (i.value or "") for i in at.info)


# ------------------------------------------------- V0.3.0 history dual-flow


def _seed_new_flow_finished_concept() -> int:
    """Create a finished V0.3.0 new-flow concept with validation + deepening data."""
    cid = database.save_concept("Opportunity Cost", "Source: choice means giving up")
    database.update_concept(
        cid,
        mastery="Understood",
        user_definition="Opportunity cost is the next-best choice I gave up",
        validation_type="definition",
        validation_task="Explain opportunity cost to a friend in one sentence",
        validation_target="Name the best alternative you gave up",
        validation_passed=True,
        validation_attempts=2,
        validation_history=json.dumps(
            [
                {
                    "answer": "First try went wrong",
                    "understanding_level": "relationship",
                    "last_response_quality": "partial",
                    "understood": [],
                    "uncertain": ["boundary"],
                    "misconceptions": [],
                },
                {
                    "answer": "Opportunity cost is the next-best choice I gave up",
                    "understanding_level": "application",
                    "last_response_quality": "deep",
                    "understood": ["core"],
                    "uncertain": [],
                    "misconceptions": [],
                },
            ],
            ensure_ascii=False,
        ),
        deeper_questions=json.dumps(
            [
                "Deeper 1: What if you never considered opportunity cost?",
                "Deeper 2: What is the relationship between opportunity cost and sunk cost?",
            ],
            ensure_ascii=False,
        ),
        deeper_answers=json.dumps(
            [
                {"question": "Deeper 1: What if you never considered opportunity cost?", "answer": "Decisions would be distorted"},
                {"question": "Deeper 2: What is the relationship between opportunity cost and sunk cost?", "answer": "Both are about choices"},
            ],
            ensure_ascii=False,
        ),
        deeper_index=2,
    )
    return cid


def test_app_history_shows_new_flow_validation_and_deepening(configured_app) -> None:
    """A new-flow concept with "My understanding": the history page should show the
    check task / answers / result and the deepening Q&A."""
    _seed_new_flow_finished_concept()
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    assert current_step(at) == "history"

    text = markdown_text(at)
    assert "Explain opportunity cost to a friend in one sentence" in text  # check task
    assert "First try went wrong" in text                                  # attempt 1
    assert "Opportunity cost is the next-best choice I gave up" in text    # attempt 2
    assert "✅ Passed" in text                                              # check result
    assert "Deeper 1: What if you never considered opportunity cost?" in text
    assert "Decisions would be distorted" in text                          # deeper answer
    assert "My answer: Both are about choices" in text


def test_app_history_shows_new_flow_unfinished_without_definition(configured_app) -> None:
    """A new-flow concept without user_definition yet: format_detail should still
    show the check/deepening records."""
    cid = database.save_concept("Opportunity Cost", "Source: choice means giving up")
    database.update_concept(
        cid,
        validation_type="definition",
        validation_task="Explain opportunity cost",
        validation_passed=False,
        validation_attempts=2,
        needs_relearning=True,
        validation_history=json.dumps(
            [{
                "answer": "Wrong answer",
                "understanding_level": "relationship",
                "last_response_quality": "partial",
                "understood": [],
                "uncertain": ["boundary"],
                "misconceptions": [],
            }],
            ensure_ascii=False,
        ),
        deeper_questions=json.dumps(["What if you never learned opportunity cost?"], ensure_ascii=False),
        deeper_answers=json.dumps([]),
        deeper_index=1,
    )
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    assert current_step(at) == "history"

    text = markdown_text(at)
    assert "Explain opportunity cost" in text
    assert "❌ Failed (3 times in a row — needs re-learning)" in text
    assert "What if you never learned opportunity cost?" in text
    assert "(unanswered)" in text


def test_app_history_legacy_qa_records_still_shown(configured_app) -> None:
    """Legacy-flow concept (no validation_type): the history page Q&A records
    still come from qa_records."""
    cid = database.save_concept("Opportunity Cost", "Source")
    database.save_qa(cid, "Does it look at the past or the future?", "The future", True)
    database.save_qa(cid, "How is it different from sunk cost?", "It looks at the past", False, hint_used=True)
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    assert current_step(at) == "history"

    text = markdown_text(at)
    assert "Does it look at the past or the future?" in text
    assert "How is it different from sunk cost?" in text
    assert "It looks at the past" in text
    assert "(hint used)" in text


# ---------------------------------------------------------------- V0.3.1 builtin content


def test_builtin_concepts_module() -> None:
    """V0.3.1 — builtin concepts module: 5 curated concepts + query interface."""
    from core.builtin_concepts import (
        BUILTIN_CONCEPTS,
        get_builtin_concept,
        get_builtin_concepts,
    )

    assert len(get_builtin_concepts()) == 5
    assert len(BUILTIN_CONCEPTS) == 5
    for c in get_builtin_concepts():
        assert c["id"] and c["title"] and c["source_text"]
        assert c["difficulty"] in (1, 2)
    c = get_builtin_concept("opportunity_cost")
    assert c is not None and c["title"] == "Opportunity Cost"
    assert "give up" in c["source_text"]
    assert get_builtin_concept("not_a_concept") is None


def test_app_home_shows_builtin_featured_concepts(configured_app) -> None:
    """V0.3.1 — "Curated concepts" cards on the home page: start learning with
    zero input."""
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert "Curated concepts" in markdown_text(at)
    for title in ("Opportunity Cost", "Compound Interest", "Survivorship Bias", "Marginal Utility", "Sunk Cost"):
        assert any(f"📖 {title}" in (b.label or "") for b in at.button)


def test_app_first_open_recommends_a_builtin(configured_app) -> None:
    """V0.3.1 — on first open (no concepts yet) one builtin is recommended
    automatically."""
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert any("Start with a curated concept" in (i.value or "") for i in at.info)


def test_app_first_open_recommend_hidden_after_learning(configured_app) -> None:
    """V0.3.1 — once there are learning records, the first-open recommendation
    disappears."""
    database.save_concept("Custom Concept", "Source")
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    assert not any("Start with a curated concept" in (i.value or "") for i in at.info)


def test_app_builtin_concept_starts_learning_without_input(configured_app) -> None:
    """V0.3.1 — clicking a curated-concept card starts learning with no input:
    the source text comes preloaded."""
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = at.button(key="builtin_opportunity_cost").click().run()
    assert not at.exception
    assert current_step(at) == "learning"
    session = at.session_state["session"]
    assert session.flow == "new"
    assert session.stage == "reading"
    assert session.title == "Opportunity Cost"
    assert "Opportunity cost" in session.source_text
    concept = database.get_concept(session.concept_id)
    assert concept["title"] == "Opportunity Cost"
    assert concept["source_text"]


def test_app_history_add_connection(configured_app) -> None:
    """V0.3.1 — adding knowledge connections moved to the history page: you can
    add a connection from inside a concept's detail."""
    a = database.save_concept("Opportunity Cost", "Source A")
    b = database.save_concept("Sunk Cost", "Source B")
    database.update_concept(a, mastery="Understood")
    at = AppTest.from_file(APP, default_timeout=15).run()
    at = click_by_label(at, "History")
    assert not at.exception
    # expand concept b's detail (a was auto-expanded on first entry, so we click b)
    at = at.button(key=f"view_{b}").click().run()
    assert not at.exception
    assert at.session_state["history_view_id"] == b

    # click "Add a knowledge connection" -> the form appears; pick a and save
    at = at.button(key=f"add_conn_{b}").click().run()
    assert not at.exception
    at.selectbox(key=f"conn_pick_{b}").select("Opportunity Cost")
    at.text_area(key=f"conn_rel_{b}").input("Both are about choices")
    at = at.button(key=f"conn_save_{b}").click().run()
    assert not at.exception
    conns = database.get_connections(b)
    assert len(conns) == 1
    assert conns[0]["concept_a_title"] == "Opportunity Cost"
    assert conns[0]["relation_text"] == "Both are about choices"
