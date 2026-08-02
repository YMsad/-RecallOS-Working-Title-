"""Database — local SQLite for points, history, and settings."""

from pathlib import Path


DB_PATH = Path(__file__).parent.parent.parent / "data" / "recallos.db"


def init_db() -> None:
    """Initialize SQLite tables."""
    raise NotImplementedError


def get_points_balance() -> int:
    """Return current AI points balance."""
    raise NotImplementedError


def deduct_points(amount: int) -> bool:
    """Deduct points. Return False if insufficient."""
    raise NotImplementedError
