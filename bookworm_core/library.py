from __future__ import annotations

import shutil
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from bookworm_core.db import Database


class LibraryService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def find_book(self, query: str) -> dict[str, Any] | None:
        books = self.db.list_books()
        if not books:
            return None

        exact = next((b for b in books if b["title"].lower() == query.lower()), None)
        if exact:
            return exact

        scored = sorted(
            books,
            key=lambda b: SequenceMatcher(
                None, query.lower(), (b["title"] or "").lower()
            ).ratio(),
            reverse=True,
        )

        best = scored[0]
        score = SequenceMatcher(None, query.lower(), best["title"].lower()).ratio()
        return best if score >= 0.55 else None

    def get_book_summary(self, query: str) -> dict[str, Any] | None:
        book = self.find_book(query)
        if not book:
            return None

        chunk_count = self.db.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE book_id = ?;", [book["id"]]
        ).fetchone()[0]

        with_pages = self.db.conn.execute(
            """
            SELECT MIN(page_number), MAX(page_number)
            FROM chunks
            WHERE book_id = ? AND page_number IS NOT NULL;
            """,
            [book["id"]],
        ).fetchone()

        chapters = self.db.conn.execute(
            """
            SELECT COUNT(DISTINCT chapter_number)
            FROM chunks
            WHERE book_id = ? AND chapter_number IS NOT NULL;
            """,
            [book["id"]],
        ).fetchone()[0]

        return {
            **book,
            "chunk_count": chunk_count,
            "page_min": with_pages[0],
            "page_max": with_pages[1],
            "chapter_count": chapters,
        }

    def get_chapter_text(self, query: str, chapter_number: int) -> tuple[dict[str, Any], str] | None:
        book = self.find_book(query)
        if not book:
            return None

        rows = self.db.get_book_chunks_for_chapter(book["id"], chapter_number)
        if not rows:
            return None

        chapter_title = next((r[0] for r in rows if r[0]), None)
        text = "\n\n".join(r[2] for r in rows)

        if chapter_title:
            text = f"{chapter_title}\n\n{text}"

        return book, text


def backup_database(db_path: Path, destination: Path) -> Path:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_dir():
        destination = destination / db_path.name

    shutil.copy2(db_path, destination)
    return destination


def restore_database(source: Path, destination_db_path: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Backup source does not exist: {source}")

    destination_db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination_db_path)
    return destination_db_path
