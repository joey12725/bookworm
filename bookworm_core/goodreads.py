from __future__ import annotations

import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from bookworm_core.db import Database


@dataclass(slots=True)
class GoodreadsImportResult:
    total_rows: int
    matched_rows: int
    unmatched_rows: int


def _norm(value: str | None) -> str:
    return (value or "").strip().lower()


class GoodreadsImporter:
    def __init__(self, db: Database) -> None:
        self.db = db

    def import_csv(
        self,
        csv_path: Path,
        *,
        min_match_score: float = 0.72,
    ) -> tuple[GoodreadsImportResult, list[dict[str, Any]]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Goodreads CSV not found: {csv_path}")

        books = self.db.list_books()
        unmatched: list[dict[str, Any]] = []
        total = 0
        matched = 0

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                title = row.get("Title") or ""
                author = row.get("Author") or ""

                best = self._best_match(books, title, author)
                if not best or best["score"] < min_match_score:
                    unmatched.append(
                        {
                            "title": title,
                            "author": author,
                            "score": best["score"] if best else None,
                        }
                    )
                    continue

                rating_raw = (row.get("My Rating") or "").strip()
                rating = float(rating_raw) if rating_raw and rating_raw != "0" else None
                read_status = (row.get("Exclusive Shelf") or "").strip() or None
                shelves = (row.get("Bookshelves") or "").strip() or None

                self.db.update_goodreads_metadata(
                    book_id=best["book"]["id"],
                    rating=rating,
                    read_status=read_status,
                    shelves=shelves,
                )
                matched += 1

        result = GoodreadsImportResult(
            total_rows=total,
            matched_rows=matched,
            unmatched_rows=total - matched,
        )
        return result, unmatched

    def _best_match(
        self, books: list[dict[str, Any]], title: str, author: str
    ) -> dict[str, Any] | None:
        if not books:
            return None

        t = _norm(title)
        a = _norm(author)

        best: dict[str, Any] | None = None
        best_score = 0.0

        for book in books:
            bt = _norm(book.get("title"))
            ba = _norm(book.get("author"))

            title_score = SequenceMatcher(None, t, bt).ratio()
            author_score = SequenceMatcher(None, a, ba).ratio() if a and ba else 0.5
            score = title_score * 0.75 + author_score * 0.25

            if score > best_score:
                best_score = score
                best = {"book": book, "score": score}

        return best
