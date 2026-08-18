"""RecallOS CLI — run a Socratic learning session or review past records.

Usage:
    python -m cli.main               # 开始一次学习
    python -m cli.main --history     # 回顾历史学习记录
"""

from __future__ import annotations

import argparse
import sys

from core import (
    DeepSeekAuthError,
    DeepSeekClient,
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
    get_connections,
    get_daily_summaries_for_concept,
    get_qa_history,
)
from core.models import MASTERY_LEARNING, MASTERY_UNCLEAR, MASTERY_UNDERSTOOD

EXIT_WORDS = {"q", "quit", "exit", "bye", "退出", "再见"}
MASTERY_LABELS = {
    MASTERY_UNDERSTOOD: "✅ Understood",
    MASTERY_UNCLEAR: "🔄 Unclear",
    MASTERY_LEARNING: "📖 Learning",
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
    title = read_line("What do you want to understand today (concept name): ")
    if title is None:
        return
    source = read_line("Paste the source text to learn (optional): ") or ""

    try:
        with DeepSeekClient() as client:
            session = LearningSession(title, source, client=client)
            question = session.start()
            while question is not None:
                print(f"\n🤔 {question}")
                answer = read_line("Your answer: ")
                if answer is None:
                    return
                result = session.submit_answer(answer)
                if result["correct"]:
                    print(f"✓ {result['feedback']}")
                else:
                    print(f"🤔 {result['feedback']}")
                    if result["hint"]:
                        print(f"💡 Hint: {result['hint']}")
                    if result["reference"]:
                        print(f"📖 Reference: {result['reference']}")
                if result["is_done"]:
                    break
                question = session.next_question()

            print("\n🔗 Suggested knowledge connections:")
            for conn in session.get_connections():
                print(f"- {conn.concept_title}: {conn.relation_text}")

            own_words = read_line("\nBiggest takeaway today (your own words, optional): ") or ""
            summary = session.finish(user_definition=own_words)
            print("\n" + "=" * 44)
            print(f"✅ I finally got it:\n{summary.breakthrough}")
            print(f"\n📌 Tomorrow's hook:\n{summary.tomorrow_hook}")
            print("=" * 44)
    except DeepSeekError as exc:
        print(f"\n❌ AI call failed: {exc}")
    except SessionError as exc:
        print(f"\n❌ Flow error: {exc}")
    except KeyboardInterrupt:
        print("\nGoodbye!")


def show_detail(concept: dict) -> None:
    """Print the full conversation for one concept: Q&A, connections, summary."""
    cid = concept["id"]
    print("\n" + "=" * 44)
    print(f"📖 {concept['title']}  {MASTERY_LABELS.get(concept['mastery'], concept['mastery'])}")
    if concept.get("user_definition"):
        print(f"My understanding: {concept['user_definition']}")
    if concept.get("source_text"):
        print(f"Source: {concept['source_text']}")

    print("\n—— Socratic Q&A ——")
    history = get_qa_history(cid)
    if not history:
        print("(none)")
    for i, qa in enumerate(history, 1):
        mark = "✓" if qa["is_correct"] else "✗"
        hint = " (used hint)" if qa["hint_used"] else ""
        print(f"{i}. Q: {qa['question']}")
        print(f"   A: {qa['user_answer']} {mark}{hint}")

    print("\n—— Knowledge connections ——")
    conns = get_connections(cid)
    if not conns:
        print("(none)")
    for conn in conns:
        print(f"- {conn['concept_a_title']} ↔ {conn['concept_b_title']}: {conn['relation_text']}")

    print("\n—— Daily summaries ——")
    summaries = get_daily_summaries_for_concept(cid)
    if not summaries:
        print("(none)")
    for s in summaries:
        if s.get("breakthrough_text"):
            print(f"I finally got it: {s['breakthrough_text']}")
        if s.get("tomorrow_hook"):
            print(f"Tomorrow's hook: {s['tomorrow_hook']}")


def _history() -> None:
    concepts = get_all_concepts()
    if not concepts:
        print("No learning records yet. Run the main mode to start a session.")
        return
    ordered = sorted(concepts, key=lambda c: _MASTERY_ORDER.index(c["mastery"]))

    print("=" * 44)
    print("📚 RecallOS History")
    print("=" * 44)
    while True:
        for i, c in enumerate(ordered, 1):
            print(f" {i:>2}. {c['title']}  {MASTERY_LABELS.get(c['mastery'], c['mastery'])}")
        choice = read_line("\nEnter a number to view details, q to quit: ")
        if choice is None:
            return
        if choice.isdigit() and 1 <= int(choice) <= len(ordered):
            show_detail(ordered[int(choice) - 1])
            print("\n" + "-" * 44)
        else:
            print("Invalid input, please enter a number from the list.")


def _ensure_api_key() -> bool:
    """Ensure an API key is configured; prompt, persist and validate if missing.

    Returns True when a key is ready, False when the user quits.
    """
    if get_settings().deepseek_api_key or get_settings().recallos_worker_url:
        return True
    while True:
        try:
            line = input("Enter your DeepSeek API Key:").strip()
        except (KeyboardInterrupt, EOFError):
            return False
        if not line:
            continue
        save_api_key_to_config(line)
        reset_settings_cache()
        try:
            with DeepSeekClient() as client:
                client.chat(
                    [{"role": "user", "content": "Please reply with exactly two characters: OK"}], max_tokens=10
                )
            return True
        except DeepSeekAuthError:
            print("Invalid key, please try again")
            continue
        except DeepSeekError:
            return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="RecallOS", description="Turn studying into thinking")
    parser.add_argument("--history", action="store_true", help="Review past learning records")
    args = parser.parse_args(argv)

    print("=" * 44)
    print("RecallOS — turn studying into thinking")
    print("Press q / quit anytime to exit")
    print("=" * 44)
    init_db()

    if args.history:
        _history()
    else:
        if not _ensure_api_key():
            return
        _learn()


if __name__ == "__main__":
    main(sys.argv[1:])
