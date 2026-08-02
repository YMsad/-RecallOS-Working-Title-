"""RecallOS CLI — run a Socratic learning session or review past records.

Usage:
    python -m cli.main               # 开始一次学习
    python -m cli.main --history     # 回顾历史学习记录
"""

from __future__ import annotations

import argparse
import sys

from core import (
    DeepSeekClient,
    DeepSeekError,
    LearningSession,
    SessionError,
    init_db,
)
from core.database import (
    get_all_concepts,
    get_connections,
    get_daily_summaries_for_concept,
    get_qa_history,
)
from core.models import MASTERY_LEARNING, MASTERY_UNCLEAR, MASTERY_UNDERSTOOD

EXIT_WORDS = {"q", "quit", "exit", "退出", "再见"}
MASTERY_LABELS = {
    MASTERY_UNDERSTOOD: "✅ 搞懂了",
    MASTERY_UNCLEAR: "🔄 模糊",
    MASTERY_LEARNING: "📖 学习中",
}
_MASTERY_ORDER = [MASTERY_UNDERSTOOD, MASTERY_UNCLEAR, MASTERY_LEARNING]


def read_line(prompt: str) -> str | None:
    """Read a line; return None when the user quits (q / Ctrl+C / Ctrl+D)."""
    try:
        line = input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if line.lower() in EXIT_WORDS:
        return None
    return line


def _learn() -> None:
    title = read_line("今天想弄懂什么（概念名）：")
    if title is None:
        return
    source = read_line("粘贴想学的原文（可留空）：") or ""

    try:
        with DeepSeekClient() as client:
            session = LearningSession(title, source, client=client)
            question = session.start()
            while question is not None:
                print(f"\n🤔 {question}")
                answer = read_line("你的回答：")
                if answer is None:
                    return
                result = session.submit_answer(answer)
                if result["correct"]:
                    print(f"✓ {result['feedback']}")
                else:
                    print(f"🤔 {result['feedback']}")
                    if result["hint"]:
                        print(f"💡 提示：{result['hint']}")
                    if result["reference"]:
                        print(f"📖 参考：{result['reference']}")
                if result["is_done"]:
                    break
                question = session.next_question()

            print("\n🔗 知识连接推荐：")
            for conn in session.get_connections():
                print(f"- {conn.concept_title}：{conn.relation_text}")

            own_words = read_line("\n今天最大的收获（用自己的话，可留空）：") or ""
            summary = session.finish(user_definition=own_words)
            print("\n" + "=" * 44)
            print(f"✅ 我终于搞懂了：\n{summary.breakthrough}")
            print(f"\n📌 明天AI会追问你：\n{summary.tomorrow_hook}")
            print("=" * 44)
    except DeepSeekError as exc:
        print(f"\n❌ AI 调用失败：{exc}")
    except SessionError as exc:
        print(f"\n❌ 流程错误：{exc}")
    except KeyboardInterrupt:
        print("\n再见！")


def show_detail(concept: dict) -> None:
    """Print the full conversation for one concept: Q&A, connections, summary."""
    cid = concept["id"]
    print("\n" + "=" * 44)
    print(f"📖 {concept['title']}  {MASTERY_LABELS.get(concept['mastery'], concept['mastery'])}")
    if concept.get("user_definition"):
        print(f"我的理解：{concept['user_definition']}")
    if concept.get("source_text"):
        print(f"来源：{concept['source_text']}")

    print("\n—— 追问记录 ——")
    history = get_qa_history(cid)
    if not history:
        print("（无）")
    for i, qa in enumerate(history, 1):
        mark = "✓" if qa["is_correct"] else "✗"
        hint = "（用过提示）" if qa["hint_used"] else ""
        print(f"{i}. Q: {qa['question']}")
        print(f"   A: {qa['user_answer']} {mark}{hint}")

    print("\n—— 知识连接 ——")
    conns = get_connections(cid)
    if not conns:
        print("（无）")
    for conn in conns:
        print(f"- {conn['concept_a_title']} ↔ {conn['concept_b_title']}：{conn['relation_text']}")

    print("\n—— 每日总结 ——")
    summaries = get_daily_summaries_for_concept(cid)
    if not summaries:
        print("（无）")
    for s in summaries:
        if s.get("breakthrough_text"):
            print(f"我终于搞懂了：{s['breakthrough_text']}")
        if s.get("tomorrow_hook"):
            print(f"明天AI会追问：{s['tomorrow_hook']}")


def _history() -> None:
    concepts = get_all_concepts()
    if not concepts:
        print("还没有学习记录。先运行主模式开始一次学习吧。")
        return
    ordered = sorted(concepts, key=lambda c: _MASTERY_ORDER.index(c["mastery"]))

    print("=" * 44)
    print("📚 RecallOS 历史回顾")
    print("=" * 44)
    while True:
        for i, c in enumerate(ordered, 1):
            print(f" {i:>2}. {c['title']}  {MASTERY_LABELS.get(c['mastery'], c['mastery'])}")
        choice = read_line("\n输入编号查看详情，q 退出：")
        if choice is None:
            return
        if choice.isdigit() and 1 <= int(choice) <= len(ordered):
            show_detail(ordered[int(choice) - 1])
            print("\n" + "-" * 44)
        else:
            print("无效输入，请输入列表中的编号。")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="RecallOS", description="把学习变成思考")
    parser.add_argument("--history", action="store_true", help="回顾历史学习记录")
    args = parser.parse_args(argv)

    print("=" * 44)
    print("RecallOS — 把学习变成思考")
    print("随时输入 q / 退出 结束")
    print("=" * 44)
    init_db()

    if args.history:
        _history()
    else:
        _learn()


if __name__ == "__main__":
    main(sys.argv[1:])
