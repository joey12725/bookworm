"""Plain data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(slots=True)
class Paragraph:
    """One paragraph of extracted text, with the page it came from when the format has pages."""

    text: str
    page: int | None = None


@dataclass(slots=True)
class Section:
    """A structural unit of a book: a chapter, or a sub-section of one.

    ``chapter`` is the 1-based chapter number the section belongs to (``None`` for front
    matter). ``path`` is the heading trail inside the chapter, outermost first.
    """

    chapter: int | None
    chapter_title: str | None
    path: tuple[str, ...] = ()
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)


@dataclass(slots=True)
class Document:
    """Everything an extractor pulls out of a file."""

    title: str
    author: str | None
    format: str
    sections: list[Section]
    pages: int | None = None

    @property
    def chapter_count(self) -> int:
        return len({s.chapter for s in self.sections if s.chapter is not None})


@dataclass(slots=True)
class Chunk:
    """A retrievable passage. Never spans two sections."""

    ord: int
    text: str
    chapter: int | None
    chapter_title: str | None
    section: str | None
    page_start: int | None
    page_end: int | None
    tokens: int
    #: Leading characters repeated from the previous chunk (overlap), skipped when
    #: reassembling a chapter's full text.
    overlap_chars: int = 0


@dataclass(slots=True)
class Book:
    id: int
    title: str
    author: str | None
    path: str
    format: str
    sha256: str
    size_bytes: int
    pages: int | None
    chapters: int
    chunks: int
    added_at: datetime
    updated_at: datetime
    rating: float | None = None
    shelf: str | None = None
    shelves: str | None = None


@dataclass(slots=True, frozen=True)
class Citation:
    """Where a passage lives: enough to find it again in the physical or digital book."""

    book: str
    author: str | None
    chapter: int | None
    chapter_title: str | None
    section: str | None
    page_start: int | None
    page_end: int | None

    def format(self) -> str:
        parts = [self.book]
        if self.chapter_title:
            parts.append(self.chapter_title)
        elif self.chapter is not None:
            parts.append(f"ch. {self.chapter}")
        if self.section:
            parts.append(self.section)
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                parts.append(f"pp. {self.page_start}-{self.page_end}")
            else:
                parts.append(f"p. {self.page_start}")
        return ", ".join(parts)


SearchMode = Literal["hybrid", "keyword", "vector"]


@dataclass(slots=True)
class Hit:
    """A ranked search result."""

    chunk_id: int
    book_id: int
    ord: int
    text: str
    score: float
    citation: Citation

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citation"]["label"] = self.citation.format()
        return data


@dataclass(slots=True)
class Answer:
    question: str
    text: str
    sources: list[Hit]
    cited: list[int]
    mode: Literal["llm", "extractive"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "mode": self.mode,
            "cited": self.cited,
            "sources": [h.to_dict() for h in self.sources],
        }


FileStatus = Literal[
    "added", "updated", "moved", "unchanged", "would-add", "would-update", "failed"
]


@dataclass(slots=True)
class FileResult:
    path: str
    status: FileStatus
    title: str | None = None
    chunks: int = 0
    reused_vectors: int = 0
    error: str | None = None


@dataclass(slots=True)
class IndexReport:
    files: list[FileResult] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def count(self, *statuses: FileStatus) -> int:
        return sum(1 for f in self.files if f.status in statuses)
