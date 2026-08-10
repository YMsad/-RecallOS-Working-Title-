"""RecallOS — Streamlit UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from core import (
    DeepSeekAuthError,
    DeepSeekError,
    LearningSession,
    SessionError,
    get_settings,
    init_db,
    reset_settings_cache,
    save_api_key_to_config,
)
from core.database import (
    get_all_concepts,
    get_all_connections,
    get_connections,
    get_daily_summaries_for_concept,
    get_qa_history,
    get_recent_concepts,
    get_setting,
    get_today_summary,
    save_connection,
    set_setting,
)
from core.models import MASTERY_LEARNING, MASTERY_UNCLEAR, MASTERY_UNDERSTOOD

MASTERY_LABELS = {
    MASTERY_UNDERSTOOD: "✅ 搞懂了",
    MASTERY_UNCLEAR: "🔄 模糊",
    MASTERY_LEARNING: "📖 学习中",
}
_MASTERY_ORDER = [MASTERY_UNDERSTOOD, MASTERY_UNCLEAR, MASTERY_LEARNING]


def reset_to_home() -> None:
    st.session_state.pop("session", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("summary_result", None)
    st.session_state.step = "home"


def go_home() -> None:
    reset_to_home()
    st.rerun()


def _navigate(target: str) -> None:
    st.session_state.step = target
    st.rerun()


# ------------------------------------------------------------------- home

def render_home() -> None:
    st.markdown("<h1 style='text-align:center;color:#6B6B6B;font-size:20px'>📚 RecallOS</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:26px'>今天想弄懂什么？</p>",
                unsafe_allow_html=True)

    title = st.text_input("概念名（如：机会成本）", placeholder="试着填一个概念")
    source = st.text_area("粘贴你想学的原文", placeholder="把课本内容或一段文字粘进来…")
    if st.button("开始", type="primary", use_container_width=True):
        if not title.strip():
            st.info("试试粘贴一段课本内容")
        else:
            try:
                session = LearningSession(title, source)
                question = session.start()
                st.session_state.session = session
                st.session_state.messages = [{"role": "assistant", "text": question}]
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


def render_learning() -> None:
    session = st.session_state.session
    answer = st.chat_input("你的回答…")
    if answer:
        st.session_state.messages.append({"role": "user", "text": answer})
        try:
            result = session.submit_answer(answer)
            if result["correct"]:
                reply = f"✓ {result['feedback']}"
            else:
                reply = f"🤔 {result['feedback']}"
                if result["hint"]:
                    reply += f"\n\n💡 提示：{result['hint']}"
                if result["reference"]:
                    reply += f"\n\n📖 参考：{result['reference']}"
            st.session_state.messages.append({"role": "assistant", "text": reply})

            if result["is_done"]:
                st.session_state.step = "connections"
            elif result["correct"] or result["reference"]:
                nxt = session.next_question()
                if nxt:
                    st.session_state.messages.append({"role": "assistant", "text": nxt})
        except DeepSeekAuthError:
            st.session_state.messages.append(
                {"role": "assistant", "text": "❌ Key 无效，请重新输入"})
        except (DeepSeekError, SessionError) as exc:
            st.session_state.messages.append({"role": "assistant", "text": f"❌ {exc}"})

    render_messages()
    if st.session_state.step == "connections":
        render_connections()


# -------------------------------------------------------------- connections

def render_connections() -> None:
    session = st.session_state.session
    st.markdown("## 🔗 发现一些知识连接")
    try:
        if not session.recommended_connections:
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
        edited = st.text_input("它们的关系是：", value=conn.relation_text, key=f"rel_{i}")
        target = next(
            (c for c in get_all_concepts() if c["title"] == conn.concept_title), None
        )
        if st.button("确认这个连接", key=f"save_{i}"):
            if target is not None:
                save_connection(
                    session.concept_id, target["id"], edited, is_user_edited=True
                )
                st.success("已保存")
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

def format_detail(concept: dict) -> str:
    lines = [f"# {concept['title']}  {MASTERY_LABELS.get(concept['mastery'], concept['mastery'])}"]
    if concept.get("user_definition"):
        lines.append(f"\n**我的理解：**{concept['user_definition']}")
    if concept.get("source_text"):
        lines.append(f"\n**来源：**{concept['source_text']}")

    lines.append("\n## 追问记录")
    history = get_qa_history(concept["id"])
    if not history:
        lines.append("（无）")
    for i, qa in enumerate(history, 1):
        mark = "✓" if qa["is_correct"] else "✗"
        hint = "（用过提示）" if qa["hint_used"] else ""
        lines.append(f"\n**Q{i}** {qa['question']}")
        lines.append(f"   {qa['user_answer']} {mark}{hint}")

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


def render_history() -> None:
    st.markdown("## 📚 我的知识")
    concepts = sorted(get_all_concepts(), key=lambda c: _MASTERY_ORDER.index(c["mastery"]))
    if not concepts:
        st.info("还没有学习记录，先去主页开始一次学习吧。")
        if st.button("返回主页"):
            go_home()
        return

    idx = st.selectbox(
        "选择概念查看详情",
        range(len(concepts)),
        format_func=lambda i: f"{concepts[i]['title']}  {MASTERY_LABELS.get(concepts[i]['mastery'], concepts[i]['mastery'])}",
    )
    st.markdown(format_detail(concepts[idx]))


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


def inject_css() -> None:
    st.markdown(
        """<style>
        div.msg-bubble-box div.msg-assistant, div.msg-assistant {
            background: #F5EFE6; padding: 10px 14px; border-radius: 12px;
            margin: 6px 0; line-height: 1.6;
        }
        div.msg-user {
            background: #EFEBE4; padding: 10px 14px; border-radius: 12px;
            margin: 6px 0 6px auto; line-height: 1.6; max-width: 85%;
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

    step = st.session_state.get("step", "home")
    if step == "learning":
        render_learning()
    elif step == "connections":
        render_connections()
    elif step == "summary":
        render_summary()
    elif step == "history":
        render_history()
    else:
        render_home()


main()
