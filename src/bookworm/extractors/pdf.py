"""PDF via pypdf (``pip install 'bookworm[pdf]'``).

Every paragraph keeps its page number, so citations point at real pages. When the PDF has
an outline (bookmarks), its top-level entries become chapters.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bookworm.errors import ExtractionError, MissingDependencyError
from bookworm.extractors.base import SectionBuilder, split_paragraphs
from bookworm.models import Document
from bookworm.text import normalize_whitespace

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def _outline_starts(reader: Any) -> dict[int, str]:
    """Top-level outline entries as {0-based page index: title}."""
    starts: dict[int, str] = {}
    try:
        outline = reader.outline
    except Exception:
        return starts
    for entry in outline:
        if isinstance(entry, list):  # nested children: sub-sections, ignored for chapters
            continue
        try:
            page_index = reader.get_destination_page_number(entry)
        except Exception:
            continue
        title = normalize_whitespace(str(getattr(entry, "title", "") or ""))
        if page_index is not None and page_index >= 0 and title:
            starts.setdefault(page_index, title)
    return starts


def extract_pdf(path: Path) -> Document:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise MissingDependencyError("PDF support", "pdf") from exc

    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError, ValueError) as exc:
        raise ExtractionError(f"Could not open PDF {path.name}: {exc}") from exc
    if reader.is_encrypted:
        raise ExtractionError(f"PDF is encrypted: {path.name}")

    meta = reader.metadata
    title = normalize_whitespace(str(meta.title)) if meta and meta.title else None
    author = normalize_whitespace(str(meta.author)) if meta and meta.author else None

    chapters = _outline_starts(reader)
    builder = SectionBuilder()
    for index, page in enumerate(reader.pages):
        if index in chapters:
            builder.start_chapter(chapters[index])
        try:
            raw = page.extract_text() or ""
        except Exception:  # a single bad page should not sink the book
            continue
        text = _HYPHEN_BREAK_RE.sub(r"\1\2", raw.replace("\r\n", "\n"))
        for paragraph in split_paragraphs(text):
            builder.add_paragraph(paragraph, page=index + 1)

    return Document(
        title=title or path.stem,
        author=author,
        format="pdf",
        sections=builder.finish(),
        pages=len(reader.pages),
    )
