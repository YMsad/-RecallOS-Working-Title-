"""AI provider — generate flashcards from text chunks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Card:
    front: str
    back: str
    source: Optional[str] = None  # e.g. "Chapter 3, Page 42"


class AIProvider(ABC):
    """Abstract base for all AI providers (cloud, local)."""

    @abstractmethod
    def generate_cards(self, text_chunk: str, chunk_index: int) -> list[Card]:
        """Return a list of Cards extracted from the text chunk."""
        ...


class OpenAIAIProvider(AIProvider):
    """Use OpenAI API (GPT-4o mini) to generate cards."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model

    def generate_cards(self, text_chunk: str, chunk_index: int) -> list[Card]:
        raise NotImplementedError


class OllamaAIProvider(AIProvider):
    """Use local Ollama model (e.g. Qwen2.5 Coder) to generate cards."""

    def __init__(self, model: str = "qwen2.5-coder:1.5b", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url

    def generate_cards(self, text_chunk: str, chunk_index: int) -> list[Card]:
        raise NotImplementedError
