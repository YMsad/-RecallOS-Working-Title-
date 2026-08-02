"""Utility helpers."""

import re
import json


def clean_json(text: str) -> str:
    """Strip markdown code fences from AI output."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    return cleaned.strip()


def safe_filename(name: str) -> str:
    """Replace illegal filesystem characters."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)
