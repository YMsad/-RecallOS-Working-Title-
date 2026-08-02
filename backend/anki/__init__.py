"""Anki generator — package flashcards into .apkg files."""

from pathlib import Path
from typing import Optional

from backend.ai import Card


def create_apkg(
    deck_name: str,
    cards: list[Card],
    output_dir: str | Path = ".",
    tags: Optional[list[str]] = None,
) -> Path:
    """Create an .apkg file from a list of Cards."""
    raise NotImplementedError
