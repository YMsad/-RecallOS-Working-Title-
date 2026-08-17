"""RecallOS — Streamlit UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os

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
    MASTERY_UNDERSTOOD: "✅ 搞懂了",
    MASTERY_UNCLEAR: "🔄 模糊",
    MASTERY_LEARNING: "📖 学习中",
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
    "reading": "📖 阅读中",
    "validation": "🧠 验证理解",
    "offer": "💬 是否深入?",
    "intervention": "💡 最小干预",
    "complete": "✅ 已完成",
    "relearn": "🔄 需要重新学习",
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

    # ---- V0.3.0 新流程恢复（Learning Loop v2）----
    session = restore_session(concept["id"])
    messages: list[dict] = []
    if session.validation_task:
        messages.append(
            {"role": "assistant", "text": f"📝 验证你的理解：\n\n{session.validation_task}"}
        )
    for entry in session.validation_history:
        messages.append({"role": "user", "text": entry.get("answer", "")})
        level = entry.get("understanding_level") or "surface"
        messages.append(
            {"role": "assistant", "text": f"✅ 已分析你的理解（层级：{level}）"}
        )
    for qa in session.deeper_history:
        messages.append({"role": "assistant", "text": qa["question"]})
        messages.append({"role": "user", "text": qa.get("answer", "")})
    if session.stage == "offer":
        messages.append({"role": "assistant", "text": "✅ 你已经理解核心概念。"})
    st.session_state.session = session
    st.session_state.messages = messages
    # 清理旧的 AI 错误/重试状态，避免恢复会话时卡在重试界面
    st.session_state.pop("v_pending_answer", None)
    st.session_state.pop("v_ai_error", None)
    st.session_state.pop("v_offer", None)
    st.session_state.step = "learning"
    st.rerun()


# ------------------------------------------------------------------- home

def render_home() -> None:
    st.markdown("<h1 style='text-align:center;color:#6B6B6B;font-size:20px'>📚 RecallOS</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:26px'>今天想弄懂什么？</p>",
                unsafe_allow_html=True)

    # V0.2.2 — 首页顶部复习入口：有到期待复习的概念时显示
    due_count = len(get_due_reviews())
    if due_count:
        if st.button(f"📝 今日复习（{due_count}）", key="review_entry",
                     type="primary", use_container_width=True):
            _navigate("review_list")

    # V0.2.0 — 零基础/有基础模式切换 + 动态开场（2 个快速问题）
    # V0.3.0 — 新流程不再需要模式切换与开场偏好设置（旧流程保留，RECALLOS_NEW_FLOW=0 时恢复）
    mode = "beginner"
    level_label = "完全没接触过"
    interest_label = "先弄懂基本意思"
    if not _NEW_FLOW:
        if st.toggle("有基础（对这个概念有一定了解）", value=False, key="mode_toggle"):
            mode = "advanced"
        with st.expander("告诉我一点你的情况，我会调整开场（可选）"):
            level_map = {"完全没接触过": "zero", "听说过一点": "some", "比较熟悉": "familiar"}
            interest_map = {
                "先弄懂基本意思": "simple",
                "想深入理解": "deep",
                "想结合生活例子": "example",
            }
            level_label = st.radio("你之前接触过这个概念吗？", list(level_map), horizontal=True)
            interest_label = st.radio("今天想怎么学？", list(interest_map), horizontal=True)

    # V0.2.3 — 预热按钮移到概念名输入框右侧（同一行，零基础模式）
    col_title, col_warm = st.columns([5, 1])
    with col_title:
        title = st.text_input("概念名（如：机会成本）", placeholder="试着填一个概念", key="home_title")
    source = st.text_area("粘贴你想学的原文", placeholder="把课本内容或一段文字粘进来…", key="home_source")

    # V0.3.0 — 新的学习目标（可选，不选默认「理解概念」）
    if _NEW_FLOW:
        goal_options = ["🧠 理解概念", "🔗 建立联系", "🛠 能实际应用", "🎓 为考试掌握"]
        st.radio(
            "你这次更想做到哪一步？（可选）",
            goal_options,
            index=0,
            horizontal=True,
            key="v_learning_goal",
        )
    with col_warm:
        if mode == "beginner" and title.strip():
            if st.button("💡 预热", key="warmup_btn"):
                try:
                    with st.spinner("AI 正在生成预热解释…"):
                        st.session_state.warmup_text = warmup_concept(title, source)
                    st.rerun()
                except DeepSeekAuthError:
                    st.error("Key 无效，请重新输入")
                except DeepSeekError as exc:
                    st.error(f"AI 调用失败：{exc}")

    if mode == "beginner" and st.session_state.get("warmup_text"):
        st.info(f"💡 {st.session_state['warmup_text']}")

    # V0.2.3 — 继续学习入口：有未完成（学习中）的概念时显示在「开始」上方
    unfinished = [
        c for c in get_all_concepts()
        if c.get("mastery") in (None, MASTERY_LEARNING)
    ]
    if unfinished:
        c = unfinished[0]
        if st.button(f"📖 继续学习：{c['title']}", key="continue_learning"):
            _resume_learning(c)

    if st.button("开始", type="primary", use_container_width=True, key="start_learning"):
        if not title.strip():
            st.info("试试粘贴一段课本内容")
        elif _NEW_FLOW:
            # V0.3.0 — 新流程：只存概念，先进入「阅读原文」阶段
            try:
                goal_options = ["🧠 理解概念", "🔗 建立联系", "🛠 能实际应用", "🎓 为考试掌握"]
                goal_map = {
                    "🧠 理解概念": "understand",
                    "🔗 建立联系": "connect",
                    "🛠 能实际应用": "apply",
                    "🎓 为考试掌握": "exam",
                }
                goal_label = st.session_state.get(
                    "v_learning_goal", goal_options[0]
                )
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
                    messages.append({"role": "assistant", "text": f"💡 {st.session_state['warmup_text']}"})
                messages.append(
                    {"role": "assistant",
                     "text": f"📖 先读原文：**{title}**\n\n读完后点下方「我读完了，开始验证」。"})
                st.session_state.messages = messages
                # 清理上一轮遗留的 AI 错误/重试状态
                st.session_state.pop("pending_action", None)
                st.session_state.pop("v_pending_answer", None)
                st.session_state.pop("v_ai_error", None)
                st.session_state.step = "learning"
                st.rerun()
            except DeepSeekAuthError:
                st.error("Key 无效，请重新输入")
            except DeepSeekError as exc:
                st.error(f"AI 调用失败：{exc}")
        else:
            # 旧流程（保留）：开场问题 + 四层追问
            try:
                with st.spinner("AI 正在思考…"):
                    session = LearningSession(
                        title,
                        source,
                        mode=mode,
                        level=level_map.get(level_label, "zero"),
                        interest=interest_map.get(interest_label, "simple"),
                    )
                    question = session.start()
                st.session_state.session = session
                messages = []
                if mode == "beginner" and st.session_state.get("warmup_text"):
                    messages.append({"role": "assistant", "text": f"💡 {st.session_state['warmup_text']}"})
                messages.append({"role": "assistant", "text": question})
                st.session_state.messages = messages
                st.session_state.step = "learning"
                st.rerun()
            except DeepSeekAuthError:
                st.error("Key 无效，请重新输入")
            except DeepSeekError as exc:
                st.error(f"AI 调用失败：{exc}")

    recent = get_recent_concepts(limit=1)
    streak = get_setting("streak", "0")
    st.divider()
    if recent:
        st.caption(f"昨天你搞懂了：{recent[0]['title']}")
    st.caption(f"连续学习：第 {streak} 天")


# ---------------------------------------------------------------- learning

def render_messages() -> None:
    """Render the conversation as styled bubbles (markdown — AppTest-safe)."""
    for m in st.session_state.messages:
        role = m["role"]
        bubble = "msg-user" if role == "user" else "msg-assistant"
        st.markdown(f'<div class="{bubble}">{m["text"]}</div>', unsafe_allow_html=True)


def _render_stage_indicator(session: LearningSession) -> None:
    """V0.3.0 — 顶部阶段指示器（仅新流程会话显示）。"""
    if getattr(session, "flow", "legacy") != "new":
        return
    label = STAGE_LABELS.get(session.stage, session.stage)
    st.info(f"当前阶段: {label}")


def render_learning() -> None:
    session = st.session_state.session
    _render_stage_indicator(session)
    if getattr(session, "flow", "legacy") == "new":
        _render_learning_new(session)
    else:
        _render_learning_old(session)


def _render_learning_old(session: LearningSession) -> None:
    # 旧流程（V0.3.0 之前）：保留四层追问 UI
    answer = st.chat_input("你的回答…")

    if answer:
        st.session_state.messages.append({"role": "user", "text": answer})
        try:
            with st.spinner("AI 正在思考…"):
                result = session.submit_answer(answer)
            # V0.2.3 — feedback 与下一个问题合并到同一条消息，避免一个回答生成两个气泡
            if result["correct"]:
                reply = f"✓ {result['feedback']}"
            else:
                reply = f"🤔 {result['feedback']}"
                if result["hint"]:
                    reply += f"\n\n💡 提示：{result['hint']}"
                if result["reference"]:
                    reply += f"\n\n📖 参考：{result['reference']}"
            if result["is_done"]:
                st.session_state.step = "connections"
            elif result["correct"] or result["reference"] or result["simplified"] or result["angle_shift"]:
                nxt = session.next_question()
                if nxt:
                    reply += f"\n\n{nxt}"
            st.session_state.messages.append({"role": "assistant", "text": reply})
        except DeepSeekAuthError:
            st.session_state.messages.append(
                {"role": "assistant", "text": "❌ Key 无效，请重新输入"})
        except (DeepSeekError, SessionError) as exc:
            st.session_state.messages.append({"role": "assistant", "text": f"❌ {exc}"})

    render_messages()

    # P0 — 「我不懂」按钮紧跟输入框，答不出来时随手可点
    if st.button("😵 我不懂，请用大白话解释一下", key="explain_btn"):
        try:
            with st.spinner("AI 正在思考…"):
                explanation = session.explain()
            reply = f"💡 我换个说法：\n\n{explanation}"
            nxt = session.next_question()
            if nxt:
                reply += f"\n\n那我们再想想这个问题：\n\n{nxt}"
            st.session_state.messages.append({"role": "assistant", "text": reply})
            st.rerun()
        except DeepSeekAuthError:
            st.session_state.messages.append(
                {"role": "assistant", "text": "❌ Key 无效，请重新输入"})
        except (DeepSeekError, SessionError) as exc:
            st.session_state.messages.append({"role": "assistant", "text": f"❌ {exc}"})

    if st.session_state.step == "connections":
        render_connections()


# V0.3.0 — AI 调用统一从「按钮回调」挪到 render 分支执行：
# 按钮只负责设置 pending_action + st.rerun()；render 开头查到待办就执行对应 AI 调用。
# 这样即使调用内部抛出任何未预期异常，也会被兜底转成可重试的气泡提示，
# 不会让页面在转圈中卡死（CLI 正常、Web 点「开始」转圈的问题由此规避）。
_PENDING_SPINNERS = {
    "start_validation": "AI 正在设计验证任务…",
    "start_validation_again": "AI 正在设计验证任务…",
    "ask_simplify": "生成大白话解释…",
    "submit_validation": "AI 正在分析你的理解…",
    "choose_deepening": "AI 正在找下一个理解缺口…",
    "submit_intervention": "AI 正在更新你的理解状态…",
}


def _run_pending(session: LearningSession) -> None:
    """执行按钮触发的、需要 AI 的待办动作（参数已放进 session_state）。"""
    action = st.session_state.get("pending_action")
    print(f"[RecallOS][_run_pending] 开始 pending_action={action!r} stage={session.stage!r} has_answer={st.session_state.get('v_pending_answer') is not None}", flush=True)
    if not action:
        return
    try:
        with st.spinner(_PENDING_SPINNERS[action]):
            if action in ("start_validation", "start_validation_again"):
                task_text = session.start_validation()
                prefix = "新一轮验证" if action == "start_validation_again" else "验证你的理解"
                st.session_state.messages.append(
                    {"role": "assistant", "text": f"📝 {prefix}：\n\n{task_text}"})
            elif action == "ask_simplify":
                explanation = session.ask_simplify()
                st.session_state.messages.append(
                    {"role": "assistant", "text": f"💡 大白话：\n\n{explanation}"})
            elif action == "submit_validation":
                answer = st.session_state.get("v_pending_answer")
                if not answer:
                    raise SessionError("缺少待分析的答案")
                result = session.submit_validation(answer)
                st.session_state.pop("v_pending_answer", None)
                if result["stage"] == "complete":
                    if result.get("final_note"):
                        st.session_state.messages.append(
                            {"role": "assistant", "text": result["final_note"]})
                elif result["stage"] == "offer":
                    st.session_state.messages.append(
                        {"role": "assistant",
                         "text": f"✅ 你已经理解核心概念（层级：{result['understanding_level']}）。"})
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
                    raise SessionError("缺少待分析的答案")
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
        print(f"[RecallOS][_run_pending][{action}] 失败：API Key 无效", flush=True)
        st.session_state.pop("pending_action", None)
        st.session_state.pop("v_pending_answer", None)
        st.session_state.messages.append({"role": "assistant", "text": "❌ Key 无效，请重新输入"})
    except (DeepSeekError, SessionError) as exc:
        print(f"[RecallOS][_run_pending][{action}] 失败(可重试)：{exc!r}", flush=True)
        st.session_state.pop("pending_action", None)
        st.session_state.messages.append(
            {"role": "assistant", "text": f"❌ {exc}\n\n再试一次即可重试。"})
    except Exception as exc:  # noqa: BLE001 —— 兜底：任何异常都转成可重试提示，杜绝转圈卡死
        logger.exception("_run_pending[%s] 意外失败", action)
        print(f"[RecallOS][_run_pending][{action}] 意外异常：{exc!r}", flush=True)
        st.session_state.pop("pending_action", None)
        st.session_state.pop("v_pending_answer", None)
        st.session_state.messages.append(
            {"role": "assistant", "text": f"❌ 出错了：{exc}，请再试一次。"})


def _capture_chat_answer(session: LearningSession) -> None:
    """在 `_run_pending` 之后捕获 chat_input 回答，并在同一 run 内立即消费。

    绝不能在这里调用 st.rerun()：streamlit#7629 中 st.chat_input 的值会在手动
    rerun 之后的 run 里被再次读到 → 同一个回答被重复提交/无限循环/界面无响应
    （现象：验证通过后同一气泡连续出现多次）。回答由本 run 内的 _run_pending
    直接处理，处理完 stage 已推进，因此页面渲染时按最新阶段画即可。
    """
    stage = session.stage
    if stage == "validation":
        placeholder, action = "合上原文，用自己的话把这件事讲清楚…", "submit_validation"
    elif stage == "intervention" and session.current_intervention() is not None:
        placeholder, action = "你的回答…", "submit_intervention"
    else:
        return
    answer = st.chat_input(placeholder, key="v_chat")
    print(f"[RecallOS][chat_input] stage={stage!r} action={action!r} answer={answer!r}", flush=True)
    if not answer:
        return
    print(f"[RecallOS][chat_input] 已捕获回答 len={len(answer)}，交给 _run_pending 同 run 处理", flush=True)
    st.session_state.messages.append({"role": "user", "text": answer})
    st.session_state["v_pending_answer"] = answer
    st.session_state["pending_action"] = action
    _run_pending(session)


def _render_pre_answer_signals(session: LearningSession) -> None:
    """在 chat_input 之前渲染可选的用户信号输入（信心预测 / 干预反馈）。

    全部可选、不阻塞：用户可以直接跳过用 chat_input 作答。
    """
    if session.stage == "validation" and session.should_ask_confidence():
        with st.container():
            st.markdown("### 🔮 猜一下")
            st.caption("被提问之前先做个预测（可选，不猜也能继续）：")
            pick = st.radio(
                "验证前的信心预测",
                ["😊 应该可以讲清楚", "🤔 不确定，容易卡住"],
                horizontal=True,
                key="v_conf_pick",
            )
            if st.button("记录预测", key="v_conf_ok"):
                session.record_confidence_prediction(
                    "clear" if pick.startswith("😊") else "unsure"
                )
                st.rerun()
    elif session.stage == "intervention" and session.feedback_pending():
        with st.container():
            st.markdown("### 💬 小反馈")
            st.caption("刚才的提示有帮到你吗？（可选，也可以直接回答）")
            pick = st.radio(
                "干预反馈",
                ["👍 清楚多了", "🤔 还是有点懵"],
                horizontal=True,
                key="v_fb_pick",
            )
            if st.button("提交反馈", key="v_fb_ok"):
                session.record_intervention_feedback(
                    "clear" if "清楚多了" in pick else "unclear"
                )
                st.rerun()


def _reading_paragraphs(source_text: str) -> list[str]:
    source_text = (source_text or "").strip()
    if not source_text:
        return []
    return [p.strip() for p in source_text.split("\n\n") if p.strip()]


def _render_stuck_signal(session: LearningSession) -> None:
    """阅读阶段：有「🤔 没看懂」标记时，出现可选的「哪里卡住了？」输入框。"""
    confused = [s for s in session.reading_signals if s.get("kind") == "confused"]
    if not confused:
        return
    pos_text = "，".join(f"第 {s['position'] + 1} 段" for s in confused[:5])
    if st.session_state.pop("v_stuck_saved", False):
        st.session_state["v_stuck_text"] = ""
    with st.expander(f"❓ 哪里卡住了？（已标 {len(confused)} 处：{pos_text}）"):
        stuck = st.text_input("用一句话说说你卡在哪（可选）", key="v_stuck_text")
        if st.button("保存", key="v_stuck_save") and stuck.strip():
            session.record_stuck_point(stuck.strip())
            st.session_state["v_stuck_saved"] = True
            st.rerun()


def _render_learning_new(session: LearningSession) -> None:
    # V0.3.0 — 新流程：阅读原文 → 验证理解 → 深入选择 → 最小干预 → 完成
    # 先执行按钮触发的 AI 待办，再捕获 chat_input 回答（同一 run 内处理，
    # 不 st.rerun()，规避 streamlit#7629），最后按最新阶段渲染。
    _run_pending(session)
    _render_pre_answer_signals(session)
    _capture_chat_answer(session)
    stage = session.stage

    if stage == "reading":
        st.markdown("### 📖 阅读原文")
        paragraphs = _reading_paragraphs(session.source_text)
        if not paragraphs:
            st.info("没有粘贴原文，直接开始验证也可以。")
        for i, para in enumerate(paragraphs):
            st.markdown(para)
            # V0.3.0 — 阅读中的理解信号（可选，不点也能继续）
            c1, c2, c3 = st.columns(3)
            if c1.button("🤔", key=f"v_rs_c_{i}", help="没看懂"):
                session.record_reading_signal("confused", i)
                st.rerun()
            if c2.button("💡", key=f"v_rs_m_{i}", help="大概懂"):
                session.record_reading_signal("match", i)
                st.rerun()
            if c3.button("✓", key=f"v_rs_k_{i}", help="我懂了"):
                session.record_reading_signal("clear", i)
                st.rerun()
        _render_stuck_signal(session)
        if session.reading_signals:
            kinds = {
                "confused": "🤔",
                "match": "💡",
                "clear": "✓",
            }
            counts = {k: sum(1 for s in session.reading_signals if s.get("kind") == k) for k in kinds}
            st.caption("已标记：" + "、".join(f"{kinds[k]} {counts[k]}" for k in kinds if counts[k]))
        if st.button("我读完了，开始验证", key="v_read_done"):
            st.session_state["pending_action"] = "start_validation"
            st.rerun()
        render_messages()
        return

    if stage == "validation":
        st.markdown("### 📝 验证你的理解")
        with st.expander("📄 再看一眼原文"):
            st.markdown(session.source_text or "（没有粘贴原文）")
        if session.validation_task:
            st.markdown(session.validation_task)

        if st.button("😵 我看不懂，帮我解释", key="v_explain_btn"):
            st.session_state["pending_action"] = "ask_simplify"
            st.rerun()

        render_messages()
        return

    if stage == "offer":
        # 深入不是默认行为：先请用户自己决定是否继续
        offer_text = st.session_state.get("v_offer")
        if offer_text is None:
            error = st.session_state.get("v_ai_error")
            if error:
                st.error(f"AI 生成深入邀请失败：{error}")
                render_messages()
                if st.button("🔄 重试", key="v_retry_offer"):
                    st.session_state.pop("v_ai_error", None)
                    st.rerun()
                return
            try:
                with st.spinner("AI 正在准备下一步…"):
                    offer = session.offer_deepening()
            except DeepSeekAuthError:
                st.session_state["v_ai_error"] = "Key 无效，请重新输入"
                st.session_state.messages.append(
                    {"role": "assistant", "text": "❌ Key 无效，请重新输入"})
                render_messages()
                st.rerun()
                return
            except (DeepSeekError, SessionError) as exc:
                st.session_state["v_ai_error"] = str(exc)
                st.session_state.messages.append({"role": "assistant", "text": f"❌ {exc}"})
                render_messages()
                st.rerun()
                return
            except Exception as exc:  # noqa: BLE001 —— 兜底：不中断页面，转为可重试
                logger.exception("深入邀请生成意外失败")
                st.session_state["v_ai_error"] = str(exc)
                st.session_state.messages.append({"role": "assistant", "text": f"❌ 出错了：{exc}"})
                render_messages()
                st.rerun()
                return
            if offer:
                st.session_state.pop("v_ai_error", None)
                st.session_state["v_offer"] = offer["offer"]
                st.session_state.messages.append(
                    {"role": "assistant", "text": f"🎯 {offer['offer']}"})
                # 不 st.rerun()：本 run 继续渲染「继续深入？」区（streamlit#7629 防护）

            offer_text = st.session_state.get("v_offer")

        if offer_text is None:
            st.success("✅ 理解这一步已经完成。")
            render_messages()
            if st.button("进入总结", type="primary", use_container_width=True, key="v_finish_offer"):
                st.session_state.pop("v_offer", None)
                _navigate("connections")
            return

        st.markdown("### 💬 继续深入？")
        with st.expander("🎯 深入邀请"):
            st.markdown(offer_text)
        c_go, c_stop = st.columns(2)
        with c_go:
            if st.button("🔍 我想再深入一层", use_container_width=True, key="v_go_deeper"):
                st.session_state.pop("v_offer", None)
                st.session_state.pop("v_pending_answer", None)
                st.session_state["pending_action"] = "choose_deepening"
                st.rerun()
        with c_stop:
            if st.button("✅ 先到这里", use_container_width=True, key="v_stop_deeper"):
                try:
                    result = session.choose_deepening(False)
                except SessionError as exc:
                    st.session_state.messages.append(
                        {"role": "assistant", "text": f"❌ {exc}"})
                else:
                    if result.get("final_note"):
                        st.session_state.messages.append(
                            {"role": "assistant", "text": result["final_note"]})
                st.session_state.pop("v_offer", None)
                st.rerun()
        render_messages()
        return

    if stage == "intervention":
        st.markdown("### 💡 最小干预")
        # 屏幕上那道干预未被回答过：AI 自动决策下一条（也用于恢复会话后自动续上）
        intervention = session.current_intervention()
        if intervention is None:
            error = st.session_state.get("v_ai_error")
            if error:
                st.error(f"AI 决策干预失败：{error}")
                render_messages()
                if st.button("🔄 重试", key="v_retry_intervention"):
                    st.session_state.pop("v_ai_error", None)
                    st.rerun()
                return
            try:
                with st.spinner("AI 正在找下一个理解缺口…"):
                    result = session.next_intervention()
            except DeepSeekAuthError:
                st.session_state["v_ai_error"] = "Key 无效，请重新输入"
                st.session_state.messages.append(
                    {"role": "assistant", "text": "❌ Key 无效，请重新输入"})
                render_messages()
                st.rerun()
                return
            except (DeepSeekError, SessionError) as exc:
                st.session_state["v_ai_error"] = str(exc)
                st.session_state.messages.append({"role": "assistant", "text": f"❌ {exc}"})
                render_messages()
                st.rerun()
                return
            except Exception as exc:  # noqa: BLE001 —— 兜底：不中断页面，转为可重试
                logger.exception("干预决策意外失败")
                st.session_state["v_ai_error"] = str(exc)
                st.session_state.messages.append({"role": "assistant", "text": f"❌ 出错了：{exc}"})
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

        st.caption("跟着提示想一想，用自己的话回答就好，不用追求标准答案。")
        render_messages()
        return

    if stage == "complete":
        st.success("🎉 这一步已经完成啦！")
        render_messages()
        if st.button("进入总结", type="primary", use_container_width=True, key="v_finish"):
            st.session_state.pop("v_deeper_question", None)
            _navigate("connections")
        return

    if stage == "relearn":
        st.error("连续 3 次没有通过验证，这个知识点建议回看原文后重新学习。")
        if st.button("重新读一遍，再试一次", key="v_retry"):
            st.session_state["pending_action"] = "start_validation_again"
            st.rerun()
        render_messages()
        return

    # 未知阶段兜底：只显示对话气泡
    render_messages()


# ------------------------------------------------------------------- review

def render_review_list() -> None:
    st.markdown("## 📝 今日复习")
    due = get_due_reviews()
    if not due:
        st.info("今天没有到期待复习的概念，先去学点新东西吧。")
        if st.button("返回主页"):
            go_home()
        return

    st.caption("以下概念到了复习时间，AI 会用上次学完时的追问来检验你。")
    for c in due:
        st.markdown(f"**{c['title']}**  {MASTERY_LABELS.get(c['mastery'], c['mastery'])}")
        if st.button("开始复习", key=f"review_{c['id']}"):
            st.session_state.review_session = ReviewSession(c["id"])
            st.session_state.review_messages = []
            st.session_state.review_finished = False
            _navigate("review")
        st.divider()
    if st.button("返回主页"):
        go_home()


def render_review() -> None:
    session = st.session_state.get("review_session")
    if session is None:
        _navigate("review_list")
        return

    st.markdown(f"## 📝 复习：{session.title}")

    answer = st.chat_input("你的回答…", key="v_chat_review")

    if not st.session_state.get("review_messages"):
        try:
            with st.spinner("AI 正在出题…"):
                question = session.start()
            st.session_state.review_messages.append(
                {"role": "assistant", "text": question})
            st.rerun()
        except DeepSeekAuthError:
            st.error("Key 无效，请重新输入")
        except (DeepSeekError, SessionError) as exc:
            st.error(f"AI 调用失败：{exc}")
        return

    if answer and not st.session_state.get("review_finished"):
        st.session_state.review_messages.append({"role": "user", "text": answer})
        try:
            with st.spinner("AI 正在判分…"):
                result = session.submit_answer(answer)
            if result["passed"]:
                reply = f"✓ {result['feedback']}"
            else:
                reply = f"🤔 {result['feedback']}"
                if result.get("needs_relearn"):
                    reply += "\n\n📖 三次都没答对，需要重新学习这个概念。"
            st.session_state.review_messages.append(
                {"role": "assistant", "text": reply})
            if session.phase == "finished":
                st.session_state.review_finished = True
        except DeepSeekAuthError:
            st.session_state.review_messages.append(
                {"role": "assistant", "text": "❌ Key 无效，请重新输入"})
        except (DeepSeekError, SessionError) as exc:
            st.session_state.review_messages.append(
                {"role": "assistant", "text": f"❌ {exc}"})
        # 不 st.rerun()：同 run 内继续渲染以下气泡（st.chat_input 提交后手动
        # rerun 会触发 streamlit#7629 —— 同一个值被再次读到、无限循环/无响应）

    for m in st.session_state.review_messages:
        role = m["role"]
        bubble = "msg-user" if role == "user" else "msg-assistant"
        st.markdown(f'<div class="{bubble}">{m["text"]}</div>', unsafe_allow_html=True)

    if st.session_state.get("review_finished"):
        if session.needs_relearn:
            st.warning("三次都没答对，这个知识点建议重新学一遍。")
        else:
            st.success("复习通过，掌握更牢固了！")
        if st.button("返回复习列表", use_container_width=True):
            _navigate("review_list")


# -------------------------------------------------------------- connections

def render_connections() -> None:
    session = st.session_state.session
    st.markdown("## 🔗 发现一些知识连接")
    try:
        if not session.recommended_connections:
            with st.spinner("AI 正在思考…"):
                session.get_connections()
    except DeepSeekAuthError:
        st.error("Key 无效，请重新输入")
        return
    except DeepSeekError as exc:
        st.error(f"连接推荐失败：{exc}")
        return

    st.caption("以下是 AI 认为和你刚学的概念有关的已学概念，可以编辑关系说明。")
    for i, conn in enumerate(session.recommended_connections):
        st.markdown(f"**{session.title}** ↔ **{conn.concept_title}**")
        edited = st.text_area("它们的关系是：", value=conn.relation_text,
                              key=f"rel_{i}", height=100)
        target = next(
            (c for c in get_all_concepts() if c["title"] == conn.concept_title), None
        )
        col_save, col_jump = st.columns(2)
        with col_save:
            if st.button("确认这个连接", key=f"save_{i}"):
                if target is not None:
                    save_connection(
                        session.concept_id, target["id"], edited, is_user_edited=True
                    )
                    st.success("已保存")
        with col_jump:
            if target is not None:
                if st.button("查看该概念详情", key=f"view_{i}"):
                    st.session_state.concept_detail_id = target["id"]
                    _navigate("concept_detail")
        st.divider()

    if st.button("进入总结", type="primary", use_container_width=True):
        st.session_state.step = "summary"
        st.rerun()


# ------------------------------------------------------------------ summary

def render_summary() -> None:
    session = st.session_state.session
    st.markdown("## ✅ 今天学完了")
    if st.session_state.get("summary_result") is None:
        own = st.text_input("我终于搞懂了……（用你自己的话，可留空）")
        if st.button("生成总结", type="primary", use_container_width=True):
            had_summary = get_today_summary() is not None
            with st.spinner("AI 正在思考…"):
                summary = session.finish(user_definition=own or "")
            if not had_summary:
                streak = int(get_setting("streak", "0")) + 1
                set_setting("streak", str(streak))
            st.session_state.summary_result = summary
            st.rerun()
        return

    summary = st.session_state.summary_result
    st.markdown(f"**我终于搞懂了：**\n\n{summary.breakthrough}")

    all_concepts = get_all_concepts()
    mastered = sum(1 for c in all_concepts if c["mastery"] == MASTERY_UNDERSTOOD)
    conn_count = len(get_all_connections())
    st.markdown("### 📊 今日收获")
    st.markdown(f"- 搞懂了 **1** 个概念")
    st.markdown(f"- 建立了 **{conn_count}** 条连接")
    st.markdown(f"- 累计掌握 **{mastered}** 个概念")

    st.markdown(f"### 📌 明天AI会追问你\n\n{summary.tomorrow_hook}")

    if st.button("明天继续", type="primary", use_container_width=True):
        go_home()


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
    """旧流程（V0.3.0 之前）的学习记录：qa_records 表里的逐层追问。"""
    lines: list[str] = []
    history = get_qa_history(concept["id"])
    if not history:
        lines.append("（无）")
    for i, qa in enumerate(history, 1):
        mark = "✓" if qa["is_correct"] else "✗"
        hint = "（用过提示）" if qa["hint_used"] else ""
        lines.append(f"\n**Q{i}** {qa['question']}")
        lines.append(f"   {qa['user_answer']} {mark}{hint}")
    return lines


def _new_flow_records_lines(concept: dict) -> list[str]:
    """V0.3.0 — 新流程（阅读→验证→深化）的学习记录。

    验证任务与历史作答存在 concepts 表的 validation_* 字段，
    深化追问存在 deeper_questions / deeper_answers 字段（JSON）。
    """
    lines: list[str] = []

    task = concept.get("validation_task")
    if task:
        lines.append(f"**验证任务：**{task}")
        for i, e in enumerate(_load_json_list(concept.get("validation_history")), 1):
            answer = e.get("answer") or e.get("missing") or ""
            level = e.get("understanding_level")
            if level is not None:
                # Learning Loop v2：答案是学习者的状态快照（无对错）
                lines.append(f"- 第 {i} 次：{answer}（层级：{level}）")
            elif e.get("passed"):
                lines.append(f"- 第 {i} 次：{answer}（✅ 通过）")
            else:
                lines.append(f"- 第 {i} 次：{answer}（❌ 未通过）")
        if concept.get("validation_passed"):
            result = "✅ 通过"
        elif concept.get("needs_relearning"):
            result = "❌ 未通过（连续 3 次，需要重新学习）"
        else:
            result = "⏳ 进行中"
        lines.append(f"**验证结果：**{result}")
    else:
        lines.append("**验证任务：**（尚未开始）")

    deeper_qs = _load_json_list(concept.get("deeper_questions"))
    if deeper_qs:
        ans_by_q = {
            e.get("question"): e.get("answer")
            for e in _load_json_list(concept.get("deeper_answers"))
        }
        lines.append("**深化追问：**")
        for i, q in enumerate(deeper_qs, 1):
            lines.append(f"- 🔍 {i}. {q}")
            lines.append(f"  我的回答：{ans_by_q.get(q) or '（未回答）'}")
    else:
        lines.append("**深化追问：**（尚未开始）")

    return lines


def _records_lines(concept: dict) -> list[str]:
    """学习记录双流程支持：有新流程标记（validation_type 非空）显示验证+深化，
    否则回退旧的 qa_records 显示。"""
    if concept.get("validation_type"):
        return _new_flow_records_lines(concept)
    return _legacy_records_lines(concept)


def format_detail(concept: dict) -> str:
    lines = [f"# {concept['title']}  {MASTERY_LABELS.get(concept['mastery'], concept['mastery'])}"]
    if concept.get("user_definition"):
        lines.append(f"\n**我的理解：**{concept['user_definition']}")
    if concept.get("source_text"):
        lines.append(f"\n**来源：**{concept['source_text']}")

    lines.append("\n## 追问记录")
    lines.extend(_records_lines(concept))

    lines.append("\n## 知识连接")
    conns = get_connections(concept["id"])
    if not conns:
        lines.append("（无）")
    for conn in conns:
        lines.append(f"- {conn['concept_a_title']} ↔ {conn['concept_b_title']}：{conn['relation_text']}")

    lines.append("\n## 每日总结")
    summaries = get_daily_summaries_for_concept(concept["id"])
    if not summaries:
        lines.append("（无）")
    for s in summaries:
        if s.get("breakthrough_text"):
            lines.append(f"- 我终于搞懂了：{s['breakthrough_text']}")
        if s.get("tomorrow_hook"):
            lines.append(f"- 明天AI会追问：{s['tomorrow_hook']}")
    return "\n".join(lines)


def render_concept_detail() -> None:
    """V0.2.2 — 概念详情页，支持从连接双向跳转。"""
    cid = st.session_state.get("concept_detail_id")
    concept = get_concept(cid) if cid else None
    if concept is None:
        st.error("这个概念不存在")
        if st.button("返回主页"):
            go_home()
        return

    st.markdown(f"## 📄 {concept['title']}")
    # 详情页辅助入口：顶部保留删除按钮，两步式确认与历史页一致
    st.button(
        "🗑 删除这个概念",
        key=f"detail_del_{concept['id']}",
        on_click=_request_delete_concept,
        args=(concept["id"],),
    )
    if st.session_state.get(f"confirm_x_{concept['id']}"):
        st.warning(
            f"确定删除「{concept['title']}」吗？"
            "该概念的所有追问、连接、复习与总结记录都会被清除，且无法恢复。"
        )
        c_ok, c_no = st.columns(2)
        with c_ok:
            st.button(
                "确认删除",
                type="primary",
                key=f"confirm_ok_{concept['id']}",
                on_click=_confirm_delete_concept,
                args=(concept["id"],),
            )
        with c_no:
            st.button(
                "取消",
                key=f"confirm_no_{concept['id']}",
                on_click=_cancel_delete_concept,
                args=(concept["id"],),
            )
    st.markdown(format_detail(concept))

    st.markdown("### 🔗 连接跳转")
    conns = get_connections(concept["id"])
    if not conns:
        st.caption("（暂无连接）")
    for conn in conns:
        other_id = conn["concept_a_id"] if conn["concept_a_id"] != concept["id"] else conn["concept_b_id"]
        other_title = conn["concept_a_title"] if conn["concept_a_title"] != concept["title"] else conn["concept_b_title"]
        if st.button(f"去往「{other_title}」", key=f"jump_{conn['id']}"):
            st.session_state.concept_detail_id = other_id
            st.rerun()

    if st.button("返回历史"):
        _navigate("history")


def _render_concept_detail_without_edit(concept: dict) -> None:
    """在没有 user_definition 时，仅正常显示概念详情（不带编辑入口）。"""
    st.markdown(format_detail(concept))

    conns = get_connections(concept["id"])
    if conns:
        st.markdown("### 🔗 从连接跳转")
        for conn in conns:
            other_id = conn["concept_a_id"] if conn["concept_a_id"] != concept["id"] else conn["concept_b_id"]
            other_title = conn["concept_a_title"] if conn["concept_a_title"] != concept["title"] else conn["concept_b_title"]
            if st.button(f"去往「{other_title}」", key=f"hist_jump_wo_{conn['id']}"):
                st.session_state.concept_detail_id = other_id
                _navigate("concept_detail")


def _render_concept_detail_with_edit_button(concept: dict) -> None:
    """在概念详情中显示「我的理解」和「编辑」入口。"""
    # 显示用户理解
    if concept.get("user_definition"):
        st.markdown(f"**我的理解：**{concept['user_definition']}")
    else:
        st.caption("（暂无理解记录）")

    # 编辑入口
    if st.button("✏️ 编辑", key=f"edit_def_{concept['id']}"):
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
    """V0.3.0 — 历史页「查看」选中的概念，详情行内展开（紧跟该概念行）。"""
    # ---- V0.2.3: 「我的理解」编辑功能（仅在有内容时显示）----
    if concept.get("user_definition"):
        edit_key = f"edit_def_{concept['id']}"
        if st.session_state.get(edit_key) is not None:
            # 编辑模式
            edited = st.text_area(
                "我的理解",
                value=st.session_state[edit_key],
                key=edit_key,
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("保存", key=f"save_{edit_key}"):
                    database.update_concept(
                        concept["id"], user_definition=edited.strip()
                    )
                    st.session_state.pop(edit_key, None)
                    st.rerun()
            with c2:
                if st.button("取消", key=f"cancel_{edit_key}"):
                    st.session_state.pop(edit_key, None)
                    st.rerun()
        else:
            # 初始状态：显示用户理解和编辑按钮
            st.markdown(f"**我的理解：**{concept['user_definition']}")
            if st.button("✏️ 编辑", key=f"edit_def_{concept['id']}"):
                st.session_state[edit_key] = concept["user_definition"]
                st.rerun()
    else:
        # 无 user_definition：仅正常显示详情（不带编辑入口）
        _render_concept_detail_without_edit(concept)
    # ---- 结束 V0.2.3 ----

    # V0.3.0 — 有「我的理解」的新流程概念在这个分支不渲染 format_detail，
    # 因此在这里补齐它的验证/深化记录显示（旧流程保持原样不变）。
    if concept.get("validation_type") and concept.get("user_definition"):
        st.markdown("### 追问记录")
        st.markdown("\n".join(_new_flow_records_lines(concept)))

    conns = get_connections(concept["id"])
    if conns:
        st.markdown("### 🔗 从连接跳转")
        for conn in conns:
            other_id = conn["concept_a_id"] if conn["concept_a_id"] != concept["id"] else conn["concept_b_id"]
            other_title = conn["concept_a_title"] if conn["concept_a_title"] != concept["title"] else conn["concept_b_title"]
            if st.button(f"去往「{other_title}」", key=f"hist_jump_{concept['id']}_{conn['id']}"):
                st.session_state.concept_detail_id = other_id
                _navigate("concept_detail")

    st.button(
        "关闭",
        key=f"close_{concept['id']}",
        on_click=_collapse_history_detail,
        args=(concept["id"],),
    )


def render_history() -> None:
    st.markdown("## 📚 我的知识")
    concepts = sorted(get_all_concepts(), key=lambda c: _MASTERY_ORDER.index(c["mastery"]))
    if not concepts:
        st.info("还没有学习记录，先去主页开始一次学习吧。")
        if st.button("返回主页"):
            go_home()
        return

    # 默认展开第一个概念（仅在首次进入历史页时；用户主动收起后不再自动展开）
    if "history_init" not in st.session_state:
        st.session_state.history_view_id = concepts[0]["id"]
        st.session_state.history_init = True

    # ---- V0.2.3: 三个独立表格，按掌握度分组（概念名称 / 操作，行内用分割线分隔）----
    for group_key in _MASTERY_ORDER:
        group = [c for c in concepts if c["mastery"] == group_key]
        if not group:
            continue
        st.markdown(f"### {MASTERY_LABELS[group_key]}")
        with st.container(border=True):
            hd_l, hd_a = st.columns([6, 4])
            with hd_l:
                st.markdown("**概念名称**")
            with hd_a:
                st.markdown("**操作**")
            st.markdown('<div class="row-divider"></div>', unsafe_allow_html=True)
            for i, c in enumerate(group):
                col_l, col_a = st.columns([6, 4])
                with col_l:
                    st.markdown(f"**{c['title']}**")
                with col_a:
                    bv, bx = st.columns(2)
                    with bv:
                        st.button(
                            "查看",
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
                # 行内两步确认：对话框紧跟被点行，避免跳到页面底部造成交互断层
                if st.session_state.get(f"confirm_x_{c['id']}"):
                    st.warning(
                        f"确定删除「{c['title']}」吗？"
                        "该概念的所有追问、连接、复习与总结记录都会被清除，且无法恢复。"
                    )
                    c_ok, c_no = st.columns(2)
                    with c_ok:
                        st.button(
                            "确认删除",
                            type="primary",
                            key=f"confirm_ok_{c['id']}",
                            on_click=_confirm_delete_concept,
                            args=(c["id"],),
                        )
                    with c_no:
                        st.button(
                            "取消",
                            key=f"confirm_no_{c['id']}",
                            on_click=_cancel_delete_concept,
                            args=(c["id"],),
                        )
                # 行内展开：被「查看」选中的概念，详情紧跟当前行下方显示
                if c["id"] == st.session_state.get("history_view_id"):
                    st.markdown('<div class="row-divider"></div>', unsafe_allow_html=True)
                    _render_history_inline_detail(c)
                if i < len(group) - 1:
                    st.markdown('<div class="row-divider"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------- main

def _api_key_configured() -> bool:
    return bool(get_settings().deepseek_api_key)


def render_api_key_setup() -> None:
    """Show a password field + save button when no API key is configured yet."""
    st.markdown("<h1 style='text-align:center;color:#6B6B6B;font-size:20px'>📚 RecallOS</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:24px'>首次使用，请配置 DeepSeek API Key</p>",
                unsafe_allow_html=True)
    st.caption("Key 只会保存在本机 ~/.recallos/config.json，不会上传。")

    api_key = st.text_input("DeepSeek API Key", type="password",
                            placeholder="sk-...", key="api_key_input")
    if st.button("保存", type="primary", use_container_width=True):
        key = (api_key or "").strip()
        if not key:
            st.error("Key 不能为空")
            return
        save_api_key_to_config(key)
        reset_settings_cache()
        st.rerun()


def render_reconfigure() -> None:
    """Dedicated page to replace the current API key. Saves, resets cache, returns home."""
    st.markdown("<h1 style='text-align:center;color:#6B6B6B;font-size:20px'>📚 RecallOS</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:24px'>🔑 重新配置 API Key</p>",
                unsafe_allow_html=True)
    st.caption("新 Key 会覆盖本机 ~/.recallos/config.json 中保存的旧 Key，保存后自动返回首页。")

    new_key = st.text_input("新的 DeepSeek API Key", type="password",
                            placeholder="sk-...", key="reconfigure_key_input")
    if st.button("保存并重新加载", type="primary", use_container_width=True):
        key = (new_key or "").strip()
        if not key:
            st.error("Key 不能为空")
            return
        save_api_key_to_config(key)
        reset_settings_cache()
        go_home()


def render_usage_stats() -> None:
    """V0.2.2 — 用量统计页：今日/本月/累计 Token 与成本 + 近 7 天趋势。"""
    st.markdown("## 📊 用量统计")

    today = get_usage_summary(since="date('now')")
    month = get_usage_summary(since="date('now','start of month')")
    total = get_usage_summary()

    c1, c2, c3 = st.columns(3)
    c1.metric("今日 Token", f"{today['total_tokens']:,}")
    c1.caption(f"调用 {today['calls']} 次 · 今日 {today['cost']:.4f} 元")
    c2.metric("本月 Token", f"{month['total_tokens']:,}")
    c2.caption(f"调用 {month['calls']} 次 · 本月 {month['cost']:.4f} 元")
    c3.metric("累计 Token", f"{total['total_tokens']:,}")
    c3.caption(f"调用 {total['calls']} 次 · 总成本 {total['cost']:.4f} 元")

    st.divider()
    st.markdown("### 近 7 天消耗趋势")
    trend = get_usage_trend(days=7)
    if not trend:
        st.info("还没有用量数据，去学一次就有啦。")
        return
    rows = [(t["day"], f"{t['total_tokens']:,}", t["calls"], f"{t['cost']:.4f}")
            for t in trend]
    st.table(pd.DataFrame(rows, columns=["日期", "Token", "调用次数", "成本(元)"]))
    st.caption("说明：成本按 DeepSeek 公开价估算（输入 ¥0.27/1M tokens，输出 ¥1.10/1M tokens）。")


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


def main() -> None:
    st.set_page_config(page_title="RecallOS", page_icon="📚", layout="centered")
    init_db()
    inject_css()

    if not _api_key_configured():
        render_api_key_setup()
        return

    with st.sidebar:
        st.markdown("### RecallOS")
        if st.button("🏠 主页"):
            go_home()
        if st.button("📚 历史回顾"):
            _navigate("history")
        if st.button("📊 用量统计"):
            _navigate("usage_stats")
        if st.button("🔑 重新配置 API Key"):
            _navigate("reconfigure")

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
