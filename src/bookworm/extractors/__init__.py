"""Format extractors. Each turns a file into a :class:`~bookworm.models.Document`.

Register your own with :func:`register`::

    from bookworm.extractors import register

    @register(".rtf")
    def extract_rtf(path: Path) -> Document: ...
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from bookworm.errors import UnsupportedFormatError
from bookworm.extractors.epub import extract_epub
from bookworm.extractors.html import extract_html
from bookworm.extractors.markdown import extract_markdown
from bookworm.extractors.mobi import extract_mobi
from bookworm.extractors.pdf import extract_pdf
from bookworm.extractors.plaintext import extract_text
from bookworm.models import Document

Extractor = Callable[[Path], Document]

_REGISTRY: dict[str, Extractor] = {
    ".txt": extract_text,
    ".md": extract_markdown,
    ".markdown": extract_markdown,
    ".html": extract_html,
    ".htm": extract_html,
    ".xhtml": extract_html,
    ".epub": extract_epub,
    ".pdf": extract_pdf,
    ".mobi": extract_mobi,
    ".azw": extract_mobi,
    ".azw3": extract_mobi,
}


def register(*extensions: str) -> Callable[[Extractor], Extractor]:
    def decorator(fn: Extractor) -> Extractor:
        for ext in extensions:
            _REGISTRY[ext.lower() if ext.startswith(".") else f".{ext.lower()}"] = fn
        return fn

    return decorator


def supported_extensions() -> frozenset[str]:
    return frozenset(_REGISTRY)


def extract(path: Path) -> Document:
    try:
        extractor = _REGISTRY[path.suffix.lower()]
    except KeyError:
        raise UnsupportedFormatError(f"Unsupported file type: {path.suffix or path.name}") from None
    return extractor(path)


__all__ = ["Extractor", "extract", "register", "supported_extensions"]
