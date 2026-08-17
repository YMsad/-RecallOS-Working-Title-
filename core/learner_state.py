"""Learner State — the heart of the V0.3.0 Learning Loop v2.

RecallOS no longer drives learning by "generate the next question"; instead it
continuously tracks the learner's actual understanding and only intervenes when
there is a real, valuable gap (minimal intervention). This module defines that
core state and its transitions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

# understanding levels, low -> high
LEVEL_ORDER: tuple[str, ...] = (
    "surface",
    "relationship",
    "application",
    "essence",
)


class UnderstandingLevel(str, Enum):
    SURFACE = "surface"
    RELATIONSHIP = "relationship"
    APPLICATION = "application"
    ESSENCE = "essence"


class ResponseQuality(str, Enum):
    DEEP = "deep"
    PARTIAL = "partial"
    SHALLOW = "shallow"


class ActionType(str, Enum):
    NONE = "none"
    HINT = "hint"
    ANALOGY = "analogy"
    EXAMPLE = "example"
    COUNTEREXAMPLE = "counterexample"
    QUESTION = "question"


def level_rank(level: str) -> int:
    """0=surface, 1=relationship, 2=application, 3=essence."""
    try:
        return LEVEL_ORDER.index(level)
    except ValueError:
        return LEVEL_ORDER.index("surface")


def one_level_above(level: str) -> str:
    """The next higher level to target for a user at ``level`` (capped)."""
    rank = min(level_rank(level) + 1, len(LEVEL_ORDER) - 1)
    return LEVEL_ORDER[rank]


class LearnerState:
    """The AI's running model of what the learner does and does not understand.

    Persisted inside the session's ``validation_history`` / ``deeper_answers``
    JSON entries so a resumed session can rebuild it.
    """

    def __init__(
        self,
        understanding_level: str = UnderstandingLevel.SURFACE.value,
        understood: list[str] | None = None,
        uncertain: list[str] | None = None,
        misconceptions: list[str] | None = None,
        last_response_quality: str = ResponseQuality.PARTIAL.value,
        next_best_action: str = ActionType.NONE.value,
        learning_goal: str = "understand",
        intervention_history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.understanding_level = understanding_level
        self.understood: list[str] = list(understood or [])
        self.uncertain: list[str] = list(uncertain or [])
        self.misconceptions: list[str] = list(misconceptions or [])
        self.last_response_quality = last_response_quality
        self.next_best_action = next_best_action
        # V0.3.1 — 用户信号输入：学习目标 + 干预效果历史
        self.learning_goal = learning_goal
        self.intervention_history: list[dict[str, Any]] = list(
            intervention_history or []
        )

    # ------------------------------------------------------------------- shape

    def to_dict(self) -> dict[str, Any]:
        return {
            "understanding_level": self.understanding_level,
            "understood": list(self.understood),
            "uncertain": list(self.uncertain),
            "misconceptions": list(self.misconceptions),
            "last_response_quality": self.last_response_quality,
            "next_best_action": self.next_best_action,
            "learning_goal": self.learning_goal,
            "intervention_history": list(self.intervention_history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LearnerState":
        if not data:
            return cls()
        return cls(
            understanding_level=str(data.get("understanding_level", "surface")),
            understood=[str(x) for x in data.get("understood") or []],
            uncertain=[str(x) for x in data.get("uncertain") or []],
            misconceptions=[str(x) for x in data.get("misconceptions") or []],
            last_response_quality=str(data.get("last_response_quality", "partial")),
            next_best_action=str(data.get("next_best_action", "none")),
            learning_goal=str(data.get("learning_goal") or "understand"),
            intervention_history=[
                dict(x) for x in data.get("intervention_history") or []
            ],
        )

    # --------------------------------------------------------------- analysis

    def update_from_analysis(self, analysis: dict[str, Any]) -> None:
        """Adopt a fresh snapshot from the analyzer/updater prompt's JSON."""
        level = str(analysis.get("understanding_level") or self.understanding_level)
        # never regress the highest observed level
        if level_rank(level) < level_rank(self.understanding_level):
            level = self.understanding_level
        self.understanding_level = level
        self.understood = [str(x) for x in analysis.get("understood") or []]
        self.uncertain = [str(x) for x in analysis.get("uncertain") or []]
        self.misconceptions = [str(x) for x in analysis.get("misconceptions") or []]
        self.last_response_quality = str(
            analysis.get("last_response_quality") or self.last_response_quality
        )
        self.next_best_action = str(analysis.get("next_best_action") or "none")

    # ------------------------------------------------------------------ policy

    def has_gap(self) -> bool:
        """True when the learner still has uncertainties or misconceptions."""
        return bool(self.uncertain or self.misconceptions)

    def should_stop(self) -> bool:
        """True when there is no further valuable gap worth intervening on."""
        if self.has_gap():
            return False
        return self.next_best_action in (ActionType.NONE.value, "")


__all__ = [
    "UnderstandingLevel",
    "ResponseQuality",
    "ActionType",
    "LEVEL_ORDER",
    "LearnerState",
    "level_rank",
    "one_level_above",
]