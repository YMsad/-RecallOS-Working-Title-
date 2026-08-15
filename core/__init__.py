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
    DEEPER_QUESTION_ORDER,
    SummaryResult,
    TextTypeResult,
    ValidateAnswerResult,
    ValidationTask,
    angle_shift_prompt,
    build_messages,
    check_answer_prompt,
    connections_prompt,
    deeper_question_prompt,
    detect_text_type_prompt,
    explain_prompt,
    opening_question_prompt,
    parse_json_response,
    question_prompt,
    reference_answer_prompt,
    review_question_prompt,
    simplify_explanation_prompt,
    simplify_question_prompt,
    summary_prompt,
    validate_answer_prompt,
    validate_response,
    validate_response_list,
    validation_task_prompt,
    warmup_prompt,
)
from core.review import (
    MAX_REVIEW_ATTEMPTS,
    ReviewSession,
    add_to_review_queue,
    get_due_reviews,
    update_review_status,
)
from core.session import (
    MAX_LAYER,
    VALIDATION_MAX_ATTEMPTS,
    LearningSession,
    SessionError,
    warmup_concept,
)

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
    "review_question_prompt",
    "build_messages",
    "parse_json_response",
    "validate_response",
    "validate_response_list",
    "CheckAnswerResult",
    "SummaryResult",
    "ConnectionSuggestion",
    "TextTypeResult",
    "ValidationTask",
    "ValidateAnswerResult",
    "detect_text_type_prompt",
    "validation_task_prompt",
    "validate_answer_prompt",
    "simplify_explanation_prompt",
    "deeper_question_prompt",
    "DEEPER_QUESTION_ORDER",
    "LearningSession",
    "SessionError",
    "warmup_concept",
    "MAX_LAYER",
    "VALIDATION_MAX_ATTEMPTS",
    "ReviewSession",
    "MAX_REVIEW_ATTEMPTS",
    "add_to_review_queue",
    "get_due_reviews",
    "update_review_status",
]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging for the whole app."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


setup_logging()
