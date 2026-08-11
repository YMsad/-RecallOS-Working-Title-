"""RecallOS core — DeepSeek API client and application settings."""

import logging

from core import database
from core.client import (
    DeepSeekAPIError,
    DeepSeekAuthError,
    DeepSeekClient,
    DeepSeekError,
    DeepSeekNetworkError,
    DeepSeekRateLimitError,
)
from core.config import (
    Settings,
    get_api_key_from_config,
    get_settings,
    reset_settings_cache,
    save_api_key_to_config,
)
from core.database import configure, init_db
from core.models import (
    MASTERY_LEARNING,
    MASTERY_UNCLEAR,
    MASTERY_UNDERSTOOD,
    MASTERY_VALUES,
    Mastery,
    Concept,
    Connection,
    DailySummary,
    QARecord,
    Setting,
)
from core.prompts import (
    SYSTEM_PROMPT,
    CheckAnswerResult,
    ConnectionSuggestion,
    SummaryResult,
    angle_shift_prompt,
    build_messages,
    check_answer_prompt,
    connections_prompt,
    explain_prompt,
    opening_question_prompt,
    parse_json_response,
    question_prompt,
    reference_answer_prompt,
    simplify_question_prompt,
    summary_prompt,
    validate_response,
    validate_response_list,
    warmup_prompt,
)
from core.session import LearningSession, SessionError, warmup_concept

__all__ = [
    "DeepSeekClient",
    "DeepSeekError",
    "DeepSeekAuthError",
    "DeepSeekRateLimitError",
    "DeepSeekAPIError",
    "DeepSeekNetworkError",
    "Settings",
    "get_settings",
    "get_api_key_from_config",
    "save_api_key_to_config",
    "reset_settings_cache",
    "database",
    "configure",
    "init_db",
    "MASTERY_LEARNING",
    "MASTERY_UNCLEAR",
    "MASTERY_UNDERSTOOD",
    "MASTERY_VALUES",
    "Mastery",
    "Concept",
    "Connection",
    "DailySummary",
    "QARecord",
    "Setting",
    "SYSTEM_PROMPT",
    "question_prompt",
    "opening_question_prompt",
    "check_answer_prompt",
    "summary_prompt",
    "connections_prompt",
    "reference_answer_prompt",
    "simplify_question_prompt",
    "angle_shift_prompt",
    "explain_prompt",
    "warmup_prompt",
    "build_messages",
    "parse_json_response",
    "validate_response",
    "validate_response_list",
    "CheckAnswerResult",
    "SummaryResult",
    "ConnectionSuggestion",
    "LearningSession",
    "SessionError",
    "warmup_concept",
]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging for the whole app."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


setup_logging()
