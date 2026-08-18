"""RecallOS — Streamlit UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core import (
    DEEPER_QUESTION_ORDER,
    DeepSeekAuthError,
    DeepSeekError,
    LearningSession,
    ReviewSession,
    SessionError,
    database,
    get_due_reviews,
    get_settings,
    init_db,
    reset_settings_cache,
    save_api_key_to_config,
    warmup_concept,
)
from core.builtin_concepts import get_builtin_concept, get_builtin_concepts
from core.database import (
    delete_concept,
    get_all_concepts,
    get_all_connections,
    get_concept,
    get_connections,
    get_daily_summaries_for_concept,
    get_qa_history,
    get_recent_concepts,
    get_setting,
    get_today_summary,
    get_usage_summary,
    get_usage_trend,
    save_connection,
    set_setting,
)
from core.models import MASTERY_LEARNING, MASTERY_UNCLEAR, MASTERY_UNDERSTOOD
from core.session import MAX_LAYER, restore_session

logger = logging.getLogger(__name__)

MASTERY_LABELS = {
    MASTERY_UNDERSTOOD: "✅ Understood",
    MASTERY_UNCLEAR: "🔄 Unclear",
    MASTERY_LEARNING: "📖 Learning",
}
_MASTERY_ORDER = [MASTERY_UNDERSTOOD, MASTERY_UNCLEAR, MASTERY_LEARNING]

# V0.3.0 — 新流程开关：RECALLOS_NEW_FLOW=0 时走旧四层追问流程（保留）
_NEW_FLOW = os.environ.get("RECALLOS_NEW_FLOW", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
STAGE_LABELS = {
    "reading": "📖 Reading",
    "validation": "🧠 Checking understanding",
    "offer": "💬 Go deeper?",
    "intervention": "💡 Minimal intervention",
    "complete": "✅ Complete",
    "relearn": "🔄 Needs re-learning",
}


def reset_to_home() -> None:
    st.session_state.pop("session", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("summary_result", None)
    st.session_state.pop("review_session", None)
    st.session_state.pop("review_messages", None)
    st.session_state.pop("review_finished", None)
    st.session_state.pop("concept_detail_id", None)
    st.session_state.pop("pending_action", None)
    st.session_state.pop("v_pending_answer", None)
    st.session_state.pop("v_ai_error", None)
    st.session_state.pop("v_deeper_question", None)
    st.session_state.pop("summary_error", None)
    st.session_state.step = "home"


def go_home() -> None:
    reset_to_home()
    st.rerun()


def _navigate(target: str) -> None:
    st.session_state.step = target
    st.rerun()


def _resume_learning(concept: dict) -> None:
    """V0.3.0 — 从数据库恢复未完成的学习会话。

    - 概念带新流程标记（stage/validation_type 非空）→ 用 ``restore_session``
      恢复完整的「阅读→验证→深化」状态与对话气泡；
    - 否则走 V0.2.3 的旧流程恢复（重建 session 与 messages）。
    """
    is_new_flow = bool(concept.get("stage") or concept.get("validation_type"))
    if not is_new_flow:
        session = LearningSession(concept["title"], concept.get("source_text") or "")
        session.concept_id = concept["id"]
        history = get_qa_history(concept["id"])
        messages: list[dict] = []
        for qa in history:
            messages.append({"role": "assistant", "text": qa["question"]})
            if qa.get("user_answer"):
                messages.append({"role": "user", "text": qa["user_answer"]})
            session.qa_history.append(
                {
                    "question": qa["question"],
                    "answer": qa.get("user_answer"),
                    "is_correct": bool(qa.get("is_correct")),
                    "hint": None,
                }
            )
        if history:
            session.layer = min(MAX_LAYER, len(history))
            session._current_question = history[-1]["question"]
        st.session_state.session = session
        st.session_state.messages = messages
        st.session_state.step = "learning"
        st.rerun()
        return

    # ---- V0.3.0 new-flow resume (Learning Loop v2) ----
    session = restore_session(concept["id"])
    messages: list[dict] = []
    if session.validation_task:
        messages.append(
            {"role": "assistant", "text": f"📝 Check your understanding:\n\n{session.validation_task}"}
        )
    for entry in session.validation_history:
        messages.append({"role": "user", "text": entry.get("answer", "")})
        level = entry.get("understanding_level") or "surface"
        messages.append(
            {"role": "assistant", "text": f"✅ Your understanding has been analyzed (level: {level})"}
        )
    for qa in session.deeper_history:
        messages.append({"role": "assistant", "text": qa["question"]})
        messages.append({"role": "user", "text": qa.get("answer", "")})
    if session.stage == "offer":
        messages.append({"role": "assistant", "text": "✅ You've understood the core concept."})
    st.session_state.session = session
    st.session_state.messages = messages
    # 清理旧的 AI 错误/重试状态，避免恢复会话时卡在重试界面
    st.session_state.pop("v_pending_answer", None)
    st.session_state.pop("v_ai_error", None)
    st.session_state.pop("v_offer", None)
    st.session_state.step = "learning"
    st.rerun()


# ------------------------------------------------------------------- home

def _start_new_session(title: str, source: str) -> None:
    """V0.3.1 - One-click start of the new flow: save the concept and enter the reading stage (no AI call in this step)."""
    goal_map = {
        "🧠 Understand a concept": "understand",
        "🔗 Build connections": "connect",
        "🛠 Apply in practice": "apply",
        "🎓 Master for exams": "exam",
    }
    goal_label = st.session_state.get("v_learning_goal", "🧠 Understand a concept")
    session = LearningSession(
        title,
        source,
        learning_goal=goal_map.get(goal_label, "understand"),
    )
    session.flow = "new"
    session.begin()
    st.session_state.session = session
    messages = []
    if st.session_state.get("warmup_text"):
        messages.append(
            {"role": "assistant", "text": f"💡 {st.session_state['warmup_text']}"}
        )
    messages.append(
        {"role": "assistant",
         "text": f"📖 First read the source: **{title}**\n\nTap \"I've finished reading\" below when you're done."}
    )
    st.session_state.messages = messages
    # 清理上一轮遗留的 AI 错误/重试状态 + 阅读导航归零
    st.session_state["v_read_index"] = 0
    st.session_state.pop("pending_action", None)
    st.session_state.pop("v_pending_answer", None)
    st.session_state.pop("v_ai_error", None)
    st.session_state.step = "learning"


def _start_builtin(concept_id: str) -> None:
    """V0.3.1 — 从内置精选概念一键开始学习（无需任何输入）。"""
    concept = get_builtin_concept(concept_id)
    if concept is None:
        return
    _start_new_session(concept["title"], concept["source_text"])


def render_home() -> None:
    st.markdown("<h1 style='text-align:center;color:#6B6B6B;font-size:20px'>📚 RecallOS</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:26px'>What do you want to understand today?</p>",
                unsafe_allow_html=True)

    # V0.2.2 - review entry at the top of the home page: shown when concepts are due
    due_count = len(get_due_reviews())
    if due_count:
        if st.button(f"📝 Review today ({due_count})", key="review_entry",
                     type="primary", use_container_width=True):
            _navigate("review_list")

    # V0.2.3 - continue-learning entry: shown when a concept is unfinished (learning)
    unfinished = [
        c for c in get_all_concepts()
        if c.get("mastery") in (None, MASTERY_LEARNING)
    ]
    if unfinished:
        c = unfinished[0]
        if st.button(f"📖 Keep learning: {c['title']}", key="continue_learning"):
            _resume_learning(c)

    st.divider()

    # V0.3.1 - auto-suggest on first open: a built-in concept so the user makes no decisions
    has_learned = bool(get_all_concepts())
    builtins = get_builtin_concepts()
    if not has_learned and builtins:
        st.info(f"👋 Start with a curated concept — **{builtins[0]['title']}**"
                f" ({builtins[0]['hook']})")

    # V0.3.1 - two-column home: curated concept cards + custom input
    col_feat, col_custom = st.columns([1, 1])
    with col_feat:
        st.markdown("### ✨ Curated concepts")
        st.caption("Click to start — no input needed")
        for c in builtins:
            st.button(
                f"📖 {c['title']}",
                key=f"builtin_{c['id']}",
                use_container_width=True,
                on_click=_start_builtin,
                args=(c["id"],),
            )
        if not builtins:
            st.caption("(No curated concepts yet)")

    with col_custom:
        st.markdown("### ✍️ Custom")
        st.caption("Want to learn your own concept? Paste some source text and start right away")
        title = st.text_input("Concept name (e.g., Opportunity Cost)", placeholder="Try filling in a concept", key="home_title")
        source = st.text_area("Paste the source text you want to learn", placeholder="Paste a textbook passage or a paragraph here...", key="home_source")

        # V0.3.0 - new learning goal (optional; defaults to \"understand a concept\" if unset)
        if _NEW_FLOW:
            goal_options = ["🧠 Understand a concept", "🔗 Build connections", "🛠 Apply in practice", "🎓 Master for exams"]
            st.radio(
                "What do you want to achieve this time? (optional)",
                goal_options,
                index=0,
                horizontal=True,
                key="v_learning_goal",
            )

        # V0.2.3 - warm-up button right next to the concept name input (same row, zero-basis mode)
        col_title, col_warm = st.columns([5, 1])
        with col_warm:
            if title.strip():
                if st.button("💡 Warm up", key="warmup_btn", use_container_width=True):
                    try:
                        with st.spinner("AI is generating a warm-up explanation..."):
                            st.session_state.warmup_text = warmup_concept(title, source)
                        st.rerun()
                    except DeepSeekAuthError:
                        st.error("Invalid API key, please re-enter it")
                    except DeepSeekError as exc:
                        st.error(f"AI call failed: {exc}")

        if st.session_state.get("warmup_text"):
            st.info(f"💡 {st.session_state['warmup_text']}")

        if st.button("Start", type="primary", use_container_width=True, key="start_learning"):
            if not title.strip():
                st.info("Try pasting a textbook passage, or tap a curated concept on the left")
            elif _NEW_FLOW:
                # V0.3.1 - new flow: only save the concept, enter the \"read the source\" stage (no AI call)
                try:
                    _start_new_session(title, source)
                    st.rerun()
                except DeepSeekAuthError:
                    st.error("Invalid API key, please re-enter it")
                except DeepSeekError as exc:
                    st.error(f"AI call failed: {exc}")
            else:
                # legacy flow (kept): opening question + four layers of questioning
                try:
                    with st.spinner("AI is thinking..."):
                        session = LearningSession(
                            title,
                            source,
                            mode="beginner",
                            level="zero",
                            interest="simple",
                        )
                        question = session.start()
                    st.session_state.session = session
                    messages = []
                    if st.session_state.get("warmup_text"):
                        messages.append({"role": "assistant", "text": f"💡 {st.session_state['warmup_text']}"})
                    messages.append({"role": "assistant", "text": question})
                    st.session_state.messages = messages
                    st.session_state.step = "learning"
                    st.rerun()
                except DeepSeekAuthError:
                    st.error("Invalid API key, please re-enter it")
                except DeepSeekError as exc:
                    st.error(f"AI call failed: {exc}")

    recent = get_recent_concepts(limit=1)
    streak = get_setting("streak", "0")
    st.divider()
    if recent:
        st.caption(f"Yesterday you understood: {recent[0]['title']}")
    st.caption(f"Learning streak: Day {streak}")


# ---------------------------------------------------------------- learning

def render_messages() -> None:
    """Render the conversation as styled bubbles (markdown — AppTest-safe)."""
    for m in st.session_state.messages:
        role = m["role"]
        bubble = "msg-user" if role == "user" else "msg-assistant"
        st.markdown(f'<div class="{bubble}">{m["text"]}</div>', unsafe_allow_html=True)


def _render_stage_indicator(session: LearningSession) -> None:
    """V0.3.0 - top stage indicator (shown only for new-flow sessions)."""
    if getattr(session, "flow", "legacy") != "new":
        return
    label = STAGE_LABELS.get(session.stage, session.stage)
    st.info(f"Stage: {label}")


def render_learning() -> None:
    session = st.session_state.session
    _render_stage_indicator(session)
    if getattr(session, "flow", "legacy") == "new":
        _render_learning_new(session)
    else:
        _render_learning_old(session)


def _render_learning_old(session: LearningSession) -> None:
    # legacy flow (before V0.3.0): keep the four-layer questioning UI
    answer = st.chat_input("Your answer...")

    if answer:
        st.session_state.messages.append({"role": "user", "text": answer})
        try:
            with st.spinner("AI is thinking..."):
                result = session.submit_answer(answer)
            # V0.2.3 - feedback and the next question merged into one message to avoid two bubbles per answer
            if result["correct"]:
                reply = f"✓ {result['feedback']}"
            else:
                reply = f"🤔 {result['feedback']}"
                if result["hint"]:
                    reply += f"\n\n💡 Hint: {result['hint']}"
                if result["reference"]:
                    reply += f"\n\n📖 Reference: {result['reference']}"
            if result["is_done"]:
                st.session_state.step = "connections"
            elif result["correct"] or result["reference"] or result["simplified"] or result["angle_shift"]:
                nxt = session.next_question()
                if nxt:
                    reply += f"\n\n{nxt}"
            st.session_state.messages.append({"role": "assistant", "text": reply})
        except DeepSeekAuthError:
            st.session_state.messages.append(
                {"role": "assistant", "text": "❌ Invalid API key, please re-enter it"})
        except (DeepSeekError, SessionError) as exc:
            st.session_state.messages.append({"role": "assistant", "text": f"❌ {exc}"})

    render_messages()

    # P0 - the \"I don't get it\" button sits right below the input, one tap away when stuck
    if st.button("😵 I don't get it — explain in plain words", key="explain_btn"):
        try:
            with st.spinner("AI is thinking..."):
                explanation = session.explain()
            reply = f"💡 Let me put it differently:\n\n{explanation}"
            nxt = session.next_question()
            if nxt:
                reply += f"\n\nNow let's think about this question again:\n\n{nxt}"
            st.session_state.messages.append({"role": "assistant", "text": reply})
            st.rerun()
        except DeepSeekAuthError:
            st.session_state.messages.append(
                {"role": "assistant", "text": "❌ Invalid API key, please re-enter it"})
        except (DeepSeekError, SessionError) as exc:
            st.session_state.messages.append({"role": "assistant", "text": f"❌ {exc}"})

    if st.session_state.step == "connections":
        render_connections()


# V0.3.0 — AI 调用统一从「按钮回调」挪到 render 分支执行：
# 按钮只负责设置 pending_action + st.rerun()；render 开头查到待办就执行对应 AI 调用。
# 这样即使调用内部抛出任何未预期异常，也会被兜底转成可重试的气泡提示，
# 不会让页面在转圈中卡死（CLI 正常、Web 点「开始」转圈的问题由此规避）。
_PENDING_SPINNERS = {
    "start_validation": "AI is designing a check task...",
    "start_validation_again": "AI is designing a check task...",
    "ask_simplify": "Generating a plain-language explanation...",
    "submit_validation": "AI is analyzing your understanding...",
    "choose_deepening": "AI is looking for your next understanding gap...",
    "submit_intervention": "AI is updating your understanding status...",
}


def _run_pending(session: LearningSession) -> None:
    """Run AI-powered actions triggered by buttons (arguments are staged in session_state)."""
    action = st.session_state.get("pending_action")
    print(f"[RecallOS][_run_pending] start pending_action={action!r} stage={session.stage!r} has_answer={st.session_state.get('v_pending_answer') is not None}", flush=True)
    if not action:
        return
    try:
        with st.spinner(_PENDING_SPINNERS[action]):
            if action in ("start_validation", "start_validation_again"):
                task_text = session.start_validation()
                prefix = "A new round of checking" if action == "start_validation_again" else "Check your understanding"
                st.session_state.messages.append(
                    {"role": "assistant", "text": f"📝 {prefix}:\n\n{task_text}"})
            elif action == "ask_simplify":
                explanation = session.ask_simplify()
                st.session_state.messages.append(
                    {"role": "assistant", "text": f"💡 In plain words:\n\n{explanation}"})
            elif action == "submit_validation":
                answer = st.session_state.get("v_pending_answer")
                if not answer:
                    raise SessionError("Missing answer to analyze")
                result = session.submit_validation(answer)
                st.session_state.pop("v_pending_answer", None)
                if result["stage"] == "complete":
                    if result.get("final_note"):
                        st.session_state.messages.append(
                            {"role": "assistant", "text": result["final_note"]})
                elif result["stage"] == "intervention":
                    st.session_state.messages.append(
                        {"role": "assistant", "text": result["bubble"]})
            elif action == "choose_deepening":
                result = session.choose_deepening(True)
                if result["stage"] == "complete":
                    if result.get("final_note"):
                        st.session_state.messages.append(
                            {"role": "assistant", "text": result["final_note"]})
                elif result["stage"] == "intervention":
                    st.session_state.messages.append(
                        {"role": "assistant", "text": result["bubble"]})
            elif action == "submit_intervention":
                answer = st.session_state.get("v_pending_answer")
                if not answer:
                    raise SessionError("Missing answer to analyze")
                result = session.submit_intervention_answer(answer)
                st.session_state.pop("v_pending_answer", None)
                if result["stage"] == "complete":
                    if result.get("final_note"):
                        st.session_state.messages.append(
                            {"role": "assistant", "text": result["final_note"]})
                elif result["stage"] == "intervention":
                    st.session_state.messages.append(
                        {"role": "assistant", "text": result["bubble"]})
        st.session_state.pop("pending_action", None)
    except DeepSeekAuthError:
        print(f"[RecallOS][_run_pending][{action}] failed: invalid API key", flush=True)
        st.session_state.pop("pending_action", None)
        st.session_state.pop("v_pending_answer", None)
        st.session_state.messages.append({"role": "assistant", "text": "❌ Invalid API key, please re-enter it"})
    except (DeepSeekError, SessionError) as exc:
        print(f"[RecallOS][_run_pending][{action}] failed (retryable): {exc!r}", flush=True)
        st.session_state.pop("pending_action", None)
        st.session_state.messages.append(
            {"role": "assistant", "text": f"❌ {exc}\n\nTap again to retry."})
    except Exception as exc:  # noqa: BLE001 - fallback: turn any exception into a retryable hint so the page never hangs spinning
        logger.exception("_run_pending[%s] unexpected failure", action)
        print(f"[RecallOS][_run_pending][{action}] unexpected exception: {exc!r}", flush=True)
        st.session_state.pop("pending_action", None)
        st.session_state.pop("v_pending_answer", None)
        st.session_state.messages.append(
            {"role": "assistant", "text": f"❌ Something went wrong: {exc}. Please try again."})


def _capture_chat_answer(session: LearningSession) -> None:
    """Capture the chat_input answer after `_run_pending` and consume it within the same run.

    Never call st.rerun() here: in streamlit#7629 the st.chat_input value is read
    again in the run after a manual rerun -> the same answer gets re-submitted /
    infinite loop / unresponsive UI (observed: the same bubble appearing repeatedly
    after validation passes). The answer is handled directly by _run_pending within
    this run, and since the stage has already advanced, the page just renders the
    latest stage.
    """
    stage = session.stage
    if stage == "validation":
        placeholder, action = "Close the source and explain this in your own words...", "submit_validation"
    elif stage == "intervention" and session.current_intervention() is not None:
        placeholder, action = "Your answer...", "submit_intervention"
    else:
        return
    answer = st.chat_input(placeholder, key="v_chat")
    print(f"[RecallOS][chat_input] stage={stage!r} action={action!r} answer={answer!r}", flush=True)
    if not answer:
        return
    print(f"[RecallOS][chat_input] captured answer len={len(answer)}, handing to _run_pending in the same run", flush=True)
    st.session_state.messages.append({"role": "user", "text": answer})
    st.session_state["v_pending_answer"] = answer
    st.session_state["pending_action"] = action
    _run_pending(session)


def _render_pre_answer_signals(session: LearningSession) -> None:
    """Render optional user signals before chat_input (confidence prediction / intervention feedback).

    All optional and non-blocking: the user can skip them and answer via chat_input.
    """
    if session.stage == "validation" and session.should_ask_confidence():
        st.markdown("### 🔮 Make a guess")
        st.caption("Make a prediction before being asked (optional — you can skip it):")
        c1, c2 = st.columns(2)
        if c1.button("😊 I can probably explain it", key="v_conf_c", use_container_width=True):
            session.record_confidence_prediction("😊 I can probably explain it")
            st.rerun()
        if c2.button("🤔 Not sure, might get stuck", key="v_conf_u", use_container_width=True):
            session.record_confidence_prediction("🤔 Not sure, might get stuck")
            st.rerun()
    elif session.stage == "intervention" and session.feedback_pending():
        st.markdown("### 💬 Quick feedback")
        st.caption("Did the hint help? (optional — you can also just answer)")
        c1, c2 = st.columns(2)
        if c1.button("👍 Much clearer", key="v_fb_clear", use_container_width=True):
            session.record_intervention_feedback("clear")
            st.rerun()
        if c2.button("🤔 Still confused", key="v_fb_unclear", use_container_width=True):
            session.record_intervention_feedback("unclear")
            st.rerun()


def _reading_paragraphs(source_text: str) -> list[str]:
    source_text = (source_text or "").strip()
    if not source_text:
        return []
    return [p.strip() for p in source_text.split("\n\n") if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split a text into sentences on CJK / English sentence-ending punctuation."""
    parts = re.split(r"(?<=[。！？!?；;.])", text)
    return [p.strip() for p in parts if p.strip()]


def _bold_keywords(text: str) -> str:
    """阅读视觉辅助：把「」内的关键短语加粗；已带 ** 的直接保留。"""
    if not text:
        return text
    if "**" in text:
        return text
    return re.sub(r"「([^「」]{2,12})」", r"**「\1」**", text)


def _extract_key_sentences(text: str, n: int) -> list[str]:
    """提取一段文字里最关键的 n 句（优先含「」/**强调**的，其次靠前的），保持原顺序。"""
    sentences = _split_sentences(text)
    if len(sentences) <= n:
        return sentences
    scored = []
    for idx, sent in enumerate(sentences):
        score = 0
        if "「" in sent or "**" in sent:
            score += 2
        score += max(0, n - idx)  # 越靠前越重要
        scored.append((score, idx, sent))
    kept = sorted(scored, key=lambda t: (t[0], -t[1]), reverse=True)[:n]
    return [s for _, _, s in sorted(kept, key=lambda t: t[1])]


def _render_learning_new(session: LearningSession) -> None:
    # V0.3.0 - new flow: read source -> check understanding -> deepen choice -> minimal intervention -> complete
    # Run button-triggered AI actions first, then capture the chat_input answer (handled
    # within the same run, no st.rerun(), avoiding streamlit#7629), then render the latest stage.
    _run_pending(session)
    _render_pre_answer_signals(session)
    _capture_chat_answer(session)
    stage = session.stage

    if stage == "reading":
        st.markdown("### 📖 Read the source")
        paragraphs = _reading_paragraphs(session.source_text)
        if not paragraphs:
            st.info("No source text pasted — you can go straight to the check.")
            if st.button("Start the check now", key="v_read_done", type="primary"):
                st.session_state["pending_action"] = "start_validation"
                st.rerun()
            return
        # V0.3.1 fix - split the reading into paragraphs + one guiding question per paragraph:
        # reading is not \"just reading\", but summarizing each paragraph in your own words
        # (recorded to reading_answers, used as context for validation and summary).
        # V0.3.1 hotfix - one paragraph at a time: v_read_index tracks the current paragraph,
        # with a progress bar + prev/next + \"I've finished reading\".
        # The \"key points only\" toggle shows only key sentences (keywords bolded), the full
        # text goes into an expander to reduce the stuttering feel.
        show_keys = st.toggle("📋 Key points only", key="v_read_show_keys")
        total = len(paragraphs)
        idx = st.session_state.get("v_read_index", 0)
        if idx >= total:
            idx = total - 1
            st.session_state["v_read_index"] = idx
        st.progress((idx + 1) / total)
        st.markdown(f"**Paragraph {idx + 1} / {total}**")
        para = paragraphs[idx]
        if show_keys and len(_split_sentences(para)) > 3:
            keys = _extract_key_sentences(para, 3)
            for k in keys:
                st.markdown(f"- {_bold_keywords(k)}")
            with st.expander("📄 Expand full text"):
                st.markdown(_bold_keywords(para))
        else:
            st.markdown(_bold_keywords(para))
        answer = st.text_input(
            "🗝️ What does this paragraph say? (summarize in one sentence in your own words)",
            key=f"v_read_ans_{idx}",
            placeholder="e.g., Opportunity cost means — choosing A means giving up B",
        )
        if answer and answer != session.reading_answer_text(idx):
            session.record_reading_answer(idx, answer)
        c_prev, c_next, _ = st.columns([1, 1, 2])
        with c_prev:
            if st.button("◀ Previous", key="v_read_prev", disabled=idx == 0):
                st.session_state["v_read_index"] = idx - 1
                st.rerun()
        with c_next:
            if st.button("Next ▶", key="v_read_next", disabled=idx >= total - 1):
                st.session_state["v_read_index"] = idx + 1
                st.rerun()
        answered = session.reading_answer_count()
        if answered > 0:
            st.caption(f"✅ {answered}/{total} paragraphs noted")
        else:
            st.caption("💡 Try saying what each paragraph is about in one sentence — it makes the check ahead easier")
        if st.button("I've finished reading", key="v_read_done", type="primary"):
            st.session_state["pending_action"] = "start_validation"
            st.rerun()
        return

    if stage == "validation":
        st.markdown("### 📝 Check your understanding")
        with st.expander("📄 Take another look at the source"):
            st.markdown(session.source_text or "(no source text pasted)")
        if session.validation_task:
            st.markdown(session.validation_task)

        if st.button("😵 I can't understand this — help me", key="v_explain_btn"):
            st.session_state["pending_action"] = "ask_simplify"
            st.rerun()

        render_messages()
        return

    if stage == "offer":
        # V0.3.1 - the offer stage was removed from the main flow (passing the check is the end).
        # Only kept for historical data: a restored old offer session is treated as complete
        # without asking \"go deeper?\".
        try:
            session.choose_deepening(False)
        except SessionError as exc:
            st.session_state.messages.append(
                {"role": "assistant", "text": f"❌ {exc}"})
        st.session_state.pop("v_offer", None)
        st.rerun()
        return

    if stage == "intervention":
        st.markdown("### 💡 Minimal intervention")
        # The on-screen intervention hasn't been answered: AI auto-decides the next one (also re-attached after session restore)
        intervention = session.current_intervention()
        if intervention is None:
            error = st.session_state.get("v_ai_error")
            if error:
                st.error(f"AI failed to decide the next intervention: {error}")
                render_messages()
                if st.button("🔄 Retry", key="v_retry_intervention"):
                    st.session_state.pop("v_ai_error", None)
                    st.rerun()
                return
            try:
                with st.spinner("AI is looking for your next understanding gap..."):
                    result = session.next_intervention()
            except DeepSeekAuthError:
                st.session_state["v_ai_error"] = "Invalid API key, please re-enter it"
                st.session_state.messages.append(
                    {"role": "assistant", "text": "❌ Invalid API key, please re-enter it"})
                render_messages()
                st.rerun()
                return
            except (DeepSeekError, SessionError) as exc:
                st.session_state["v_ai_error"] = str(exc)
                st.session_state.messages.append({"role": "assistant", "text": f"❌ {exc}"})
                render_messages()
                st.rerun()
                return
            except Exception as exc:  # noqa: BLE001 - fallback: don't break the page, make it retryable
                logger.exception("intervention decision unexpectedly failed")
                st.session_state["v_ai_error"] = str(exc)
                st.session_state.messages.append({"role": "assistant", "text": f"❌ Something went wrong: {exc}"})
                render_messages()
                st.rerun()
                return
            if result["stage"] == "complete":
                if result.get("final_note"):
                    st.session_state.messages.append(
                        {"role": "assistant", "text": result["final_note"]})
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "text": result["bubble"]})
            st.session_state.pop("v_ai_error", None)
            st.rerun()
            return

        st.caption("Think along with the hint and answer in your own words — there's no need to aim for a textbook answer.")
        render_messages()
        return

    if stage == "complete":
        st.success("🎉 Your understanding is confirmed — you're done for today!")
        st.caption("Finishing is just the beginning — I'll remind you to come back tomorrow for review.")
        if st.button("View today's summary", type="primary", use_container_width=True, key="v_finish"):
            st.session_state.pop("v_deeper_question", None)
            _navigate("summary")
        return

    if stage == "relearn":
        st.error("You failed the check 3 times in a row — we suggest re-reading the source and learning this again.")
        if st.button("Re-read it and try again", key="v_retry"):
            st.session_state["pending_action"] = "start_validation_again"
            st.rerun()
        render_messages()
        return

    # fallback for unknown stages: show the conversation bubbles only
    render_messages()


# ------------------------------------------------------------------- review

def render_review_list() -> None:
    st.markdown("## 📝 Review today")
    due = get_due_reviews()
    if not due:
        st.info("No concepts are due for review today — go learn something new.")
        if st.button("Back to home"):
            go_home()
        return

    st.caption("These concepts are due for review — the AI will quiz you with the questions from last time.")
    for c in due:
        st.markdown(f"**{c['title']}**  {MASTERY_LABELS.get(c['mastery'], c['mastery'])}")
        if st.button("Start review", key=f"review_{c['id']}"):
            st.session_state.review_session = ReviewSession(c["id"])
            st.session_state.review_messages = []
            st.session_state.review_finished = False
            _navigate("review")
        st.divider()
    if st.button("Back to home"):
        go_home()


def render_review() -> None:
    session = st.session_state.get("review_session")
    if session is None:
        _navigate("review_list")
        return

    st.markdown(f"## 📝 Review: {session.title}")

    answer = st.chat_input("Your answer...", key="v_chat_review")

    if not st.session_state.get("review_messages"):
        try:
            with st.spinner("AI is preparing a question..."):
                question = session.start()
            st.session_state.review_messages.append(
                {"role": "assistant", "text": question})
            st.rerun()
        except DeepSeekAuthError:
            st.error("Invalid API key, please re-enter it")
        except (DeepSeekError, SessionError) as exc:
            st.error(f"AI call failed: {exc}")
        return

    if answer and not st.session_state.get("review_finished"):
        st.session_state.review_messages.append({"role": "user", "text": answer})
        try:
            with st.spinner("AI is grading..."):
                result = session.submit_answer(answer)
            if result["passed"]:
                reply = f"✓ {result['feedback']}"
            else:
                reply = f"🤔 {result['feedback']}"
                if result.get("needs_relearn"):
                    reply += "\n\n📖 You missed it three times — this concept needs to be re-learned."
            st.session_state.review_messages.append(
                {"role": "assistant", "text": reply})
            if session.phase == "finished":
                st.session_state.review_finished = True
        except DeepSeekAuthError:
            st.session_state.review_messages.append(
                {"role": "assistant", "text": "❌ Invalid API key, please re-enter it"})
        except (DeepSeekError, SessionError) as exc:
            st.session_state.review_messages.append(
                {"role": "assistant", "text": f"❌ {exc}"})
        # no st.rerun(): keep rendering the bubbles below within the same run (a manual
        # rerun after chat_input submission triggers streamlit#7629 - the same value gets
        # read again, causing infinite loops / unresponsive UI)

    for m in st.session_state.review_messages:
        role = m["role"]
        bubble = "msg-user" if role == "user" else "msg-assistant"
        st.markdown(f'<div class="{bubble}">{m["text"]}</div>', unsafe_allow_html=True)

    if st.session_state.get("review_finished"):
        if session.needs_relearn:
            st.warning("You missed it three times — we suggest re-learning this concept.")
        else:
            st.success("Review passed — your grasp is stronger now!")
        if st.button("Back to review list", use_container_width=True):
            _navigate("review_list")


# -------------------------------------------------------------- connections

def render_connections() -> None:
    session = st.session_state.session
    st.markdown("## 🔗 Discover knowledge connections")
    try:
        if not session.recommended_connections:
            with st.spinner("AI is thinking..."):
                session.get_connections()
    except DeepSeekAuthError:
        st.error("Invalid API key, please re-enter it")
        return
    except DeepSeekError as exc:
        st.error(f"Failed to recommend connections: {exc}")
        return

    st.caption("Here are concepts the AI thinks are related to what you just learned. You can edit the relationship description.")
    for i, conn in enumerate(session.recommended_connections):
        st.markdown(f"**{session.title}** ↔ **{conn.concept_title}**")
        edited = st.text_area("What's the relationship between them:", value=conn.relation_text,
                              key=f"rel_{i}", height=100)
        target = next(
            (c for c in get_all_concepts() if c["title"] == conn.concept_title), None
        )
        col_save, col_jump = st.columns(2)
        with col_save:
            if st.button("Confirm connection", key=f"save_{i}"):
                if target is not None:
                    save_connection(
                        session.concept_id, target["id"], edited, is_user_edited=True
                    )
                    st.success("Saved")
        with col_jump:
            if target is not None:
                if st.button("View concept details", key=f"view_{i}"):
                    st.session_state.concept_detail_id = target["id"]
                    _navigate("concept_detail")
        st.divider()

    if st.button("Go to summary", type="primary", use_container_width=True):
        st.session_state.step = "summary"
        st.rerun()


# ------------------------------------------------------------------ summary

def render_summary() -> None:
    session = st.session_state.session
    st.markdown("## ✅ Done for today")
    if st.session_state.get("summary_result") is None:
        # V0.3.1 - new flow auto-generates the summary on completion, no user input needed
        if getattr(session, "flow", "legacy") == "new":
            error = st.session_state.get("summary_error")
            if error:
                st.error(f"Failed to generate summary: {error}")
                if st.button("🔄 Retry", key="summary_retry"):
                    st.session_state.pop("summary_error", None)
                    st.rerun()
                return
            try:
                had_summary = get_today_summary() is not None
                with st.spinner("AI is organizing today's takeaways..."):
                    summary = session.finish_auto()
                if not had_summary:
                    streak = int(get_setting("streak", "0")) + 1
                    set_setting("streak", str(streak))
                st.session_state.summary_result = summary
                st.rerun()
            except DeepSeekAuthError:
                st.session_state["summary_error"] = "Invalid API key, please re-enter it"
                st.rerun()
            except (DeepSeekError, SessionError) as exc:
                st.session_state["summary_error"] = str(exc)
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - fallback: make it retryable
                logger.exception("auto summary generation unexpectedly failed")
                st.session_state["summary_error"] = str(exc)
                st.rerun()
            return

        # legacy flow (kept): type your own words first, then generate
        own = st.text_input("I finally understood... (in your own words, optional)")
        if st.button("Generate summary", type="primary", use_container_width=True):
            had_summary = get_today_summary() is not None
            with st.spinner("AI is thinking..."):
                summary = session.finish(user_definition=own or "")
            if not had_summary:
                streak = int(get_setting("streak", "0")) + 1
                set_setting("streak", str(streak))
            st.session_state.summary_result = summary
            st.rerun()
        return

    summary = st.session_state.summary_result

    # V0.3.1 fix - 3-sentence summary
    st.markdown("### ✨ What you understood")
    st.markdown(f"> {summary.breakthrough}")
    st.divider()
    if getattr(summary, "plain", None):
        st.markdown("### 💡 In plain words, it means")
        st.markdown(f"> {summary.plain}")
        st.divider()
    st.markdown("### 📌 Tomorrow the AI will ask you")
    st.markdown(f"> {summary.tomorrow_hook}")
    st.divider()

    all_concepts = get_all_concepts()
    mastered = sum(1 for c in all_concepts if c["mastery"] == MASTERY_UNDERSTOOD)
    conn_count = len(get_all_connections())
    st.caption(f"📊 {mastered} concepts mastered, {conn_count} connections built")

    if st.button("👋 See you tomorrow", type="primary", use_container_width=True):
        go_home()

    # optional items (in an open collapsible, don't block leaving)
    with st.expander("🔍 Want to dig deeper? (optional)"):
        st.caption("If there's anything you haven't fully worked out, you can keep digging.")
        if st.button("Keep digging deeper", key="summary_deeper"):
            session.stage = "intervention"
            st.session_state.step = "learning"
            st.rerun()

    with st.expander("🔗 View related concepts (optional)"):
        conns = get_connections(session.concept_id) if session.concept_id else []
        if conns:
            for conn in conns:
                other_title = (
                    conn["concept_a_title"]
                    if conn["concept_a_title"] != session.title
                    else conn["concept_b_title"]
                )
                st.markdown(f"- Relationship with **{other_title}**: {conn['relation_text']}")
        else:
            st.caption("(No connections yet — you can add some on the history page)")


# ------------------------------------------------------------------ history

def _load_json_list(raw) -> list:
    """Parse a JSON-list column value; return [] for empty/invalid content."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _legacy_records_lines(concept: dict) -> list[str]:
    """Legacy flow (before V0.3.0) learning records: the layered questions in qa_records."""
    lines: list[str] = []
    history = get_qa_history(concept["id"])
    if not history:
        lines.append("(None)")
    for i, qa in enumerate(history, 1):
        mark = "✓" if qa["is_correct"] else "✗"
        hint = " (hint used)" if qa["hint_used"] else ""
        lines.append(f"\n**Q{i}** {qa['question']}")
        lines.append(f"   {qa['user_answer']} {mark}{hint}")
    return lines


def _new_flow_records_lines(concept: dict) -> list[str]:
    """V0.3.0 - new flow (read -> check -> deepen) learning records.

    The check task and answer history live in the validation_* fields of the concepts
    table; deeper questions live in deeper_questions / deeper_answers (JSON).
    """
    lines: list[str] = []

    task = concept.get("validation_task")
    if task:
        lines.append(f"**Check task:** {task}")
        for i, e in enumerate(_load_json_list(concept.get("validation_history")), 1):
            answer = e.get("answer") or e.get("missing") or ""
            level = e.get("understanding_level")
            if level is not None:
                # Learning Loop v2: the answer is a snapshot of the learner's state (no right/wrong)
                lines.append(f"- Attempt {i}: {answer} (level: {level})")
            elif e.get("passed"):
                lines.append(f"- Attempt {i}: {answer} (✅ passed)")
            else:
                lines.append(f"- Attempt {i}: {answer} (❌ failed)")
        if concept.get("validation_passed"):
            result = "✅ Passed"
        elif concept.get("needs_relearning"):
            result = "❌ Failed (3 times in a row — needs re-learning)"
        else:
            result = "⏳ In progress"
        lines.append(f"**Check result:** {result}")
    else:
        lines.append("**Check task:** (not started yet)")

    deeper_qs = _load_json_list(concept.get("deeper_questions"))
    if deeper_qs:
        ans_by_q = {
            e.get("question"): e.get("answer")
            for e in _load_json_list(concept.get("deeper_answers"))
        }
        lines.append("**Deeper questions:**")
        for i, q in enumerate(deeper_qs, 1):
            lines.append(f"- 🔍 {i}. {q}")
            lines.append(f"   My answer: {ans_by_q.get(q) or '(unanswered)'}")
    else:
        lines.append("**Deeper questions:** (not started yet)")

    return lines


def _records_lines(concept: dict) -> list[str]:
    """Learning records with dual-flow support: new-flow markers (validation_type set) show
    check + deepen; otherwise fall back to the old qa_records view."""
    if concept.get("validation_type"):
        return _new_flow_records_lines(concept)
    return _legacy_records_lines(concept)


def format_detail(concept: dict) -> str:
    lines = [f"# {concept['title']}  {MASTERY_LABELS.get(concept['mastery'], concept['mastery'])}"]
    if concept.get("user_definition"):
        lines.append(f"\n**My understanding:** {concept['user_definition']}")
    if concept.get("source_text"):
        lines.append(f"\n**Source:** {concept['source_text']}")

    lines.append("\n## Question history")
    lines.extend(_records_lines(concept))

    lines.append("\n## Knowledge connections")
    conns = get_connections(concept["id"])
    if not conns:
        lines.append("(None)")
    for conn in conns:
        lines.append(f"- {conn['concept_a_title']} ↔ {conn['concept_b_title']}: {conn['relation_text']}")

    lines.append("\n## Daily summaries")
    summaries = get_daily_summaries_for_concept(concept["id"])
    if not summaries:
        lines.append("(None)")
    for s in summaries:
        if s.get("breakthrough_text"):
            lines.append(f"- Finally understood: {s['breakthrough_text']}")
        if s.get("tomorrow_hook"):
            lines.append(f"- Tomorrow's AI question: {s['tomorrow_hook']}")
    return "\n".join(lines)


def render_concept_detail() -> None:
    """V0.2.2 - concept detail page, supports bidirectional jumps from connections."""
    cid = st.session_state.get("concept_detail_id")
    concept = get_concept(cid) if cid else None
    if concept is None:
        st.error("This concept doesn't exist")
        if st.button("Back to home"):
            go_home()
        return

    st.markdown(f"## 📄 {concept['title']}")
    # detail page helper entry: delete button at the top, two-step confirmation like the history page
    st.button(
        "🗑 Delete this concept",
        key=f"detail_del_{concept['id']}",
        on_click=_request_delete_concept,
        args=(concept["id"],),
    )
    if st.session_state.get(f"confirm_x_{concept['id']}"):
        st.warning(
            f"Are you sure you want to delete \"{concept['title']}\"? "
            "All of this concept's questions, connections, reviews and summaries will be deleted and can't be recovered."
        )
        c_ok, c_no = st.columns(2)
        with c_ok:
            st.button(
                "Delete",
                type="primary",
                key=f"confirm_ok_{concept['id']}",
                on_click=_confirm_delete_concept,
                args=(concept["id"],),
            )
        with c_no:
            st.button(
                "Cancel",
                key=f"confirm_no_{concept['id']}",
                on_click=_cancel_delete_concept,
                args=(concept["id"],),
            )
    st.markdown(format_detail(concept))

    st.markdown("### 🔗 Jump to connections")
    conns = get_connections(concept["id"])
    if not conns:
        st.caption("(No connections yet)")
    for conn in conns:
        other_id = conn["concept_a_id"] if conn["concept_a_id"] != concept["id"] else conn["concept_b_id"]
        other_title = conn["concept_a_title"] if conn["concept_a_title"] != concept["title"] else conn["concept_b_title"]
        if st.button(f"Go to \"{other_title}\"", key=f"jump_{conn['id']}"):
            st.session_state.concept_detail_id = other_id
            st.rerun()

    if st.button("Back to history"):
        _navigate("history")


def _render_concept_detail_without_edit(concept: dict) -> None:
    """When there is no user_definition, just show the concept detail normally (no edit entry)."""
    st.markdown(format_detail(concept))

    conns = get_connections(concept["id"])
    if conns:
        st.markdown("### 🔗 Jump to connections")
        for conn in conns:
            other_id = conn["concept_a_id"] if conn["concept_a_id"] != concept["id"] else conn["concept_b_id"]
            other_title = conn["concept_a_title"] if conn["concept_a_title"] != concept["title"] else conn["concept_b_title"]
            if st.button(f"Go to \"{other_title}\"", key=f"hist_jump_wo_{conn['id']}"):
                st.session_state.concept_detail_id = other_id
                _navigate("concept_detail")


def _render_concept_detail_with_edit_button(concept: dict) -> None:
    """Show \"My understanding\" and an \"Edit\" entry in the concept detail."""
    # show the user's understanding
    if concept.get("user_definition"):
        st.markdown(f"**My understanding:** {concept['user_definition']}")
    else:
        st.caption("(No understanding recorded yet)")

    # edit entry
    if st.button("✏️ Edit", key=f"edit_def_{concept['id']}"):
        st.session_state[f"edit_def_{concept['id']}"] = concept.get("user_definition", "")


# ---- 历史页按钮使用 on_click 回调（事件由服务端处理，不依赖按钮返回值的 rerun 触发）----
def _select_history_concept(cid: int) -> None:
    # 再次点击当前展开的概念即收起；一次只展开一个
    if st.session_state.get("history_view_id") == cid:
        st.session_state.history_view_id = None
    else:
        st.session_state.history_view_id = cid


def _collapse_history_detail(cid: int) -> None:
    if st.session_state.get("history_view_id") == cid:
        st.session_state.history_view_id = None


def _request_delete_concept(cid: int) -> None:
    st.session_state[f"confirm_x_{cid}"] = True


def _cancel_delete_concept(cid: int) -> None:
    st.session_state.pop(f"confirm_x_{cid}", None)


def _confirm_delete_concept(cid: int) -> None:
    delete_concept(cid)
    st.session_state.pop(f"confirm_x_{cid}", None)
    if st.session_state.get("history_view_id") == cid:
        st.session_state.history_view_id = None
    if st.session_state.get("step") == "concept_detail":
        # 详情页删除后回到历史页（回调内不能调用 st.rerun，靠回调后自动 rerun）
        st.session_state.pop("concept_detail_id", None)
        st.session_state.step = "history"


def _render_history_inline_detail(concept: dict) -> None:
    """V0.3.0 - inline-expanded detail for the concept selected with \"View\" on the history page (right under that row)."""
    # ---- V0.2.3: \"My understanding\" edit feature (shown only when there's content) ----
    if concept.get("user_definition"):
        edit_key = f"edit_def_{concept['id']}"
        if st.session_state.get(edit_key) is not None:
            # edit mode
            edited = st.text_area(
                "My understanding",
                value=st.session_state[edit_key],
                key=edit_key,
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save", key=f"save_{edit_key}"):
                    database.update_concept(
                        concept["id"], user_definition=edited.strip()
                    )
                    st.session_state.pop(edit_key, None)
                    st.rerun()
            with c2:
                if st.button("Cancel", key=f"cancel_{edit_key}"):
                    st.session_state.pop(edit_key, None)
                    st.rerun()
        else:
            # initial state: show the user's understanding and an edit button
            st.markdown(f"**My understanding:** {concept['user_definition']}")
            if st.button("✏️ Edit", key=f"edit_def_{concept['id']}"):
                st.session_state[edit_key] = concept["user_definition"]
                st.rerun()
    else:
        # no user_definition: just show the detail normally (no edit entry)
        _render_concept_detail_without_edit(concept)
    # ---- end V0.2.3 ----

    # V0.3.0 - new-flow concepts with \"My understanding\" don't render format_detail in this
    # branch, so add its check/deepen records here (legacy flow stays unchanged).
    if concept.get("validation_type") and concept.get("user_definition"):
        st.markdown("### Question history")
        st.markdown("\n".join(_new_flow_records_lines(concept)))

    conns = get_connections(concept["id"])
    st.markdown("### 🔗 Knowledge connections")
    if conns:
        for conn in conns:
            other_id = conn["concept_a_id"] if conn["concept_a_id"] != concept["id"] else conn["concept_b_id"]
            other_title = conn["concept_a_title"] if conn["concept_a_title"] != concept["title"] else conn["concept_b_title"]
            st.markdown(f"**{other_title}**: {conn['relation_text']}")
            if st.button(f"Go to \"{other_title}\"", key=f"hist_jump_{concept['id']}_{conn['id']}"):
                st.session_state.concept_detail_id = other_id
                _navigate("concept_detail")
    else:
        st.caption("(No connections yet — add some below)")

    # V0.3.1 - connections moved to the history page: actively add knowledge connections after learning (optional)
    if st.button("➕ Add a knowledge connection", key=f"add_conn_{concept['id']}"):
        st.session_state[f"show_conn_{concept['id']}"] = True
    if st.session_state.get(f"show_conn_{concept['id']}"):
        others = [c for c in get_all_concepts() if c["id"] != concept["id"]]
        if not others:
            st.caption("There are no other concepts to connect yet — go learn one on the home page.")
        else:
            titles = {c["title"]: c["id"] for c in others}
            pick = st.selectbox(
                "Connect to which concept?", list(titles), key=f"conn_pick_{concept['id']}"
            )
            rel = st.text_area(
                "How are they related?",
                key=f"conn_rel_{concept['id']}",
                height=80,
                placeholder="e.g., Opportunity cost and sunk cost are both about \"choices\"",
            )
            if st.button("Save connection", key=f"conn_save_{concept['id']}"):
                if rel.strip():
                    save_connection(
                        concept["id"], titles[pick], rel.strip(), is_user_edited=True
                    )
                    st.session_state.pop(f"show_conn_{concept['id']}", None)
                    st.success("Saved")
                    st.rerun()

    st.button(
        "Close",
        key=f"close_{concept['id']}",
        on_click=_collapse_history_detail,
        args=(concept["id"],),
    )


def render_history() -> None:
    st.markdown("## 📚 My knowledge")
    concepts = sorted(get_all_concepts(), key=lambda c: _MASTERY_ORDER.index(c["mastery"]))
    if not concepts:
        st.info("No learning records yet — go start a session on the home page.")
        if st.button("Back to home"):
            go_home()
        return

    # expand the first concept by default (only on first entry; once collapsed by the user it won't auto-expand again)
    if "history_init" not in st.session_state:
        st.session_state.history_view_id = concepts[0]["id"]
        st.session_state.history_init = True

    # ---- V0.2.3: three separate tables grouped by mastery (concept name / actions, rows separated by dividers) ----
    for group_key in _MASTERY_ORDER:
        group = [c for c in concepts if c["mastery"] == group_key]
        if not group:
            continue
        st.markdown(f"### {MASTERY_LABELS[group_key]}")
        with st.container(border=True):
            hd_l, hd_a = st.columns([6, 4])
            with hd_l:
                st.markdown("**Concept**")
            with hd_a:
                st.markdown("**Actions**")
            st.markdown('<div class="row-divider"></div>', unsafe_allow_html=True)
            for i, c in enumerate(group):
                col_l, col_a = st.columns([6, 4])
                with col_l:
                    st.markdown(f"**{c['title']}**")
                with col_a:
                    bv, bx = st.columns(2)
                    with bv:
                        st.button(
                            "View",
                            key=f"view_{c['id']}",
                            on_click=_select_history_concept,
                            args=(c["id"],),
                        )
                    with bx:
                        st.button(
                            "✕",
                            key=f"xdel_{c['id']}",
                            on_click=_request_delete_concept,
                            args=(c["id"],),
                        )
                # inline two-step confirmation: the dialog follows the clicked row so it doesn't jump to the page bottom
                if st.session_state.get(f"confirm_x_{c['id']}"):
                    st.warning(
                        f"Are you sure you want to delete \"{c['title']}\"? "
                        "All of this concept's questions, connections, reviews and summaries will be deleted and can't be recovered."
                    )
                    c_ok, c_no = st.columns(2)
                    with c_ok:
                        st.button(
                            "Delete",
                            type="primary",
                            key=f"confirm_ok_{c['id']}",
                            on_click=_confirm_delete_concept,
                            args=(c["id"],),
                        )
                    with c_no:
                        st.button(
                            "Cancel",
                            key=f"confirm_no_{c['id']}",
                            on_click=_cancel_delete_concept,
                            args=(c["id"],),
                        )
                # inline expand: the detail of the concept selected with \"View\" renders right below its row
                if c["id"] == st.session_state.get("history_view_id"):
                    st.markdown('<div class="row-divider"></div>', unsafe_allow_html=True)
                    _render_history_inline_detail(c)
                if i < len(group) - 1:
                    st.markdown('<div class="row-divider"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------- main

def _api_key_configured() -> bool:
    """已配置手动 Key，或已配置 Worker 分发（两者满足其一即可开始学习）。"""
    settings = get_settings()
    return bool(settings.deepseek_api_key) or bool(settings.recallos_worker_url)


def render_api_key_setup() -> None:
    """Show a password field + save button when no API key is configured yet."""
    st.markdown("<h1 style='text-align:center;color:#6B6B6B;font-size:20px'>📚 RecallOS</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:24px'>First time here? Configure your DeepSeek API Key</p>",
                unsafe_allow_html=True)
    st.caption("Your key is stored only locally in ~/.recallos/config.json and never uploaded.")

    api_key = st.text_input("DeepSeek API Key", type="password",
                            placeholder="sk-...", key="api_key_input")
    if st.button("Save", type="primary", use_container_width=True):
        key = (api_key or "").strip()
        if not key:
            st.error("API key cannot be empty")
            return
        save_api_key_to_config(key)
        reset_settings_cache()
        st.rerun()


def render_reconfigure() -> None:
    """Dedicated page to replace the current API key. Saves, resets cache, returns home."""
    st.markdown("<h1 style='text-align:center;color:#6B6B6B;font-size:20px'>📚 RecallOS</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:24px'>🔑 Reconfigure API Key</p>",
                unsafe_allow_html=True)
    st.caption("Your new key will overwrite the old one in ~/.recallos/config.json, then you'll be returned to the home page.")

    new_key = st.text_input("New DeepSeek API Key", type="password",
                            placeholder="sk-...", key="reconfigure_key_input")
    if st.button("Save and reload", type="primary", use_container_width=True):
        key = (new_key or "").strip()
        if not key:
            st.error("API key cannot be empty")
            return
        save_api_key_to_config(key)
        reset_settings_cache()
        go_home()


def render_usage_stats() -> None:
    """V0.2.2 - usage stats page: today/month/total tokens and cost + last 7 days trend."""
    st.markdown("## 📊 Usage statistics")

    today = get_usage_summary(since="date('now')")
    month = get_usage_summary(since="date('now','start of month')")
    total = get_usage_summary()

    c1, c2, c3 = st.columns(3)
    c1.metric("Today's tokens", f"{today['total_tokens']:,}")
    c1.caption(f"{today['calls']} calls · ¥{today['cost']:.4f} today")
    c2.metric("This month's tokens", f"{month['total_tokens']:,}")
    c2.caption(f"{month['calls']} calls · ¥{month['cost']:.4f} this month")
    c3.metric("Total tokens", f"{total['total_tokens']:,}")
    c3.caption(f"{total['calls']} calls · ¥{total['cost']:.4f} total")

    st.divider()
    st.markdown("### Last 7 days trend")
    trend = get_usage_trend(days=7)
    if not trend:
        st.info("No usage data yet — go learn something and it'll show up.")
        return
    rows = [(t["day"], f"{t['total_tokens']:,}", t["calls"], f"{t['cost']:.4f}")
            for t in trend]
    st.table(pd.DataFrame(rows, columns=["Date", "Tokens", "Calls", "Cost (¥)"]))
    st.caption("Note: cost is estimated at DeepSeek's public rates (input ¥0.27/1M tokens, output ¥1.10/1M tokens).")


def inject_css() -> None:
    st.markdown(
        """<style>
        /* V0.2.3 — 全局字体 +2px（默认 14px → 16px） */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            font-size: 16px;
        }
        [data-testid="stMarkdownContainer"], [data-testid="stText"],
        [data-testid="stCaptionContainer"] {
            font-size: 16px;
        }
        .stTextInput input, .stTextArea textarea, .stChatInput textarea,
        .stButton button, .stSelectbox > div, .stRadio label, .stCheckbox label,
        .stToggle label, .stCaption {
            font-size: 16px !important;
        }
        div.msg-bubble-box div.msg-assistant, div.msg-assistant {
            background: #F5EFE6; padding: 10px 14px; border-radius: 12px;
            margin: 6px 0; line-height: 1.6;
        }
        div.msg-user {
            background: #EFEBE4; padding: 10px 14px; border-radius: 12px;
            margin: 6px 0 6px auto; line-height: 1.6; max-width: 85%;
        }
        /* V0.2.3 — 历史页掌握度表格紧凑化：Excel 风格行高，低留白 */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 2px 8px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
            gap: 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stHorizontalBlock"] {
            gap: 8px;
            padding: 1px 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] {
            line-height: 1.2;
            margin: 0;
        }
        .row-divider {
            border-top: 1px solid rgba(128, 128, 128, 0.35);
            height: 0;
            margin: 1px 0;
        }
        </style>""",
        unsafe_allow_html=True,
    )


def _save_feedback(content: str) -> None:
    """Append one feedback entry to ~/.recallos/feedback.log."""
    content = (content or "").strip()
    if not content:
        st.warning("Please write something before submitting.")
        return
    log_path = Path.home() / ".recallos" / "feedback.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{timestamp} | {content}\n")
    st.success("✅ Thanks! I'll improve soon.")
    st.session_state.show_feedback = False
    st.session_state.pop("feedback_input", None)


def main() -> None:
    st.set_page_config(page_title="RecallOS", page_icon="📚", layout="centered")
    init_db()
    inject_css()

    if not _api_key_configured():
        render_api_key_setup()
        return

    with st.sidebar:
        st.markdown("### RecallOS")
        if st.button("🏠 Home"):
            go_home()
        if st.button("📚 History"):
            _navigate("history")
        if st.button("📊 Usage"):
            _navigate("usage_stats")
        if st.button("🔑 Reconfigure API Key"):
            _navigate("reconfigure")

        st.divider()
        if st.button("💬 Feedback / Suggestions"):
            st.session_state.show_feedback = not st.session_state.get("show_feedback", False)
        if st.session_state.get("show_feedback", False):
            feedback_text = st.text_area(
                "What's getting in the way? Any suggestions?",
                key="feedback_input",
                placeholder="Tell me where you got stuck, or how I can improve…",
            )
            col_submit, col_cancel = st.columns(2)
            with col_submit:
                if st.button("Submit Feedback", use_container_width=True):
                    _save_feedback(feedback_text)
            with col_cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_feedback = False
                    st.session_state.pop("feedback_input", None)
                    st.rerun()

    step = st.session_state.get("step", "home")
    if step == "learning":
        render_learning()
    elif step == "connections":
        render_connections()
    elif step == "summary":
        render_summary()
    elif step == "review_list":
        render_review_list()
    elif step == "review":
        render_review()
    elif step == "concept_detail":
        render_concept_detail()
    elif step == "history":
        render_history()
    elif step == "usage_stats":
        render_usage_stats()
    elif step == "reconfigure":
        render_reconfigure()
    else:
        render_home()


main()
