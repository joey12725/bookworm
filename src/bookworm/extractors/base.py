"""Building blocks shared by the format extractors."""

from __future__ import annotations

from pathlib import Path

from bookworm.models import Paragraph, Section
from bookworm.text import normalize_whitespace


def read_text_file(path: Path) -> str:
    """Read a text file as UTF-8, falling back to cp1252 for older files."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - cp1252 decodes nearly anything
        text = raw.decode("latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines and re-flow hard-wrapped lines inside each paragraph."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(normalize_whitespace(" ".join(current)))
            current = []
    if current:
        paragraphs.append(normalize_whitespace(" ".join(current)))
    return paragraphs


class SectionBuilder:
    """Accumulates paragraphs into chapters and sub-sections as an extractor walks a book.

    Chapter numbers are assigned lazily, so a heading with no text under it (a part title,
    a stray table-of-contents entry) does not consume a number.
    """

    def __init__(self) -> None:
        self._sections: list[Section] = []
        self._chapter: int | None = None
        self._chapter_title: str | None = None
        self._pending_chapter = False
        self._dirty = False
        self._path: tuple[str, ...] = ()
        self._chapter_count = 0
        self._current = Section(chapter=None, chapter_title=None)
        self._sections.append(self._current)

    @property
    def chapter_count(self) -> int:
        return self._chapter_count

    def _materialize(self) -> None:
        if self._pending_chapter:
            self._pending_chapter = False
            self._chapter_count += 1
            self._chapter = self._chapter_count
        self._current = Section(
            chapter=self._chapter, chapter_title=self._chapter_title, path=self._path
        )
        self._sections.append(self._current)

    def start_chapter(self, title: str | None) -> None:
        self._chapter_title = normalize_whitespace(title) if title else None
        self._pending_chapter = True
        self._path = ()
        self._dirty = True

    def start_subsection(self, title: str, depth: int = 1) -> None:
        self._path = (*self._path[: max(0, depth - 1)], normalize_whitespace(title))
        self._dirty = True

    def add_paragraph(self, text: str, page: int | None = None) -> None:
        text = text.strip()
        if not text:
            return
        if self._dirty:
            self._dirty = False
            self._materialize()
        self._current.paragraphs.append(Paragraph(text=text, page=page))

    def finish(self) -> list[Section]:
        return [s for s in self._sections if s.paragraphs]
