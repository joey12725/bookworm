"""Enrich books with ratings and shelves from a Goodreads library export (CSV)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from bookworm.models import Book
from bookworm.store import Store
from bookworm.text import fold, normalize_whitespace

MIN_SCORE = 0.72


@dataclass(slots=True)
class GoodreadsReport:
    rows: int = 0
    matched: list[tuple[str, str]] = field(default_factory=list)  # (csv title, library title)
    unmatched: list[str] = field(default_factory=list)


def _clean_title(title: str) -> str:
    # Goodreads appends series info: "The Hobbit (Middle-earth, #1)"
    return fold(normalize_whitespace(title.split(" (")[0].split(":")[0]))


def match_score(csv_title: str, csv_author: str, book: Book) -> float:
    title_score = SequenceMatcher(None, _clean_title(csv_title), _clean_title(book.title)).ratio()
    if csv_author and book.author:
        author_score = SequenceMatcher(None, fold(csv_author), fold(book.author)).ratio()
    else:
        author_score = 0.5
    return 0.75 * title_score + 0.25 * author_score


def import_goodreads(
    store: Store, csv_path: Path, *, min_score: float = MIN_SCORE
) -> GoodreadsReport:
    books = store.list_books()
    report = GoodreadsReport()
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            report.rows += 1
            title = (row.get("Title") or "").strip()
            author = (row.get("Author") or "").strip()
            best = max(books, key=lambda b: match_score(title, author, b), default=None)
            if best is None or match_score(title, author, best) < min_score:
                report.unmatched.append(title)
                continue
            rating_raw = (row.get("My Rating") or "").strip()
            store.update_book(
                best.id,
                rating=float(rating_raw) if rating_raw not in ("", "0") else None,
                shelf=(row.get("Exclusive Shelf") or "").strip() or None,
                shelves=(row.get("Bookshelves") or "").strip() or None,
            )
            report.matched.append((title, best.title))
    return report
