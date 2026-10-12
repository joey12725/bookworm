from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class TextFragment:
    content: str
    chapter_title: str | None = None
    chapter_number: int | None = None
    page_number: int | None = None


@dataclass(slots=True)
class ChunkRecord:
    content: str
    chapter_title: str | None
    chapter_number: int | None
    page_number: int | None
    chunk_index: int
    token_count: int


@dataclass(slots=True)
class BookRecord:
    id: str
    title: str
    author: str | None
    file_path: str
    file_hash: str
    format: str
    file_size_bytes: int
    total_pages: int | None
    goodreads_rating: float | None
    goodreads_read_status: str | None
    goodreads_shelves: str | None
    added_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class SearchResult:
    book_title: str
    author: str | None
    chapter_title: str | None
    chapter_number: int | None
    page_number: int | None
    chunk_index: int
    content: str
    score: float


@dataclass(slots=True)
class ScanResult:
    scanned_files: int
    ingested_files: int
    skipped_files: int
    failed_files: int


@dataclass(slots=True)
class AskResult:
    answer: str
    sources: list[SearchResult]
