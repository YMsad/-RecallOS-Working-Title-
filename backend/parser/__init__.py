"""Document parsers — extract text from PDF, TXT, EPUB."""

from pathlib import Path
from typing import Optional


SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".epub"}


def parse_pdf(file_path: str | Path) -> str:
    """Extract text from a PDF file."""
    raise NotImplementedError


def parse_txt(file_path: str | Path) -> str:
    """Extract text from a plain text file (auto-detect encoding)."""
    raise NotImplementedError


def parse_epub(file_path: str | Path) -> str:
    """Extract text from an EPUB file."""
    raise NotImplementedError


def parse(file_path: str | Path) -> Optional[str]:
    """Route to the correct parser based on file extension."""
    ext = Path(file_path).suffix.lower()
    parsers = {
        ".pdf": parse_pdf,
        ".txt": parse_txt,
        ".epub": parse_epub,
    }
    parser = parsers.get(ext)
    if parser is None:
        return None
    return parser(file_path)
