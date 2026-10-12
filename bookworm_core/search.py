from __future__ import annotations

from bookworm_core.db import Database
from bookworm_core.embeddings import EmbeddingService
from bookworm_core.models import SearchResult


class SearchService:
    def __init__(self, db: Database, embeddings: EmbeddingService) -> None:
        self.db = db
        self.embeddings = embeddings

    @staticmethod
    def _rows_to_results(rows: list[tuple]) -> list[SearchResult]:
        return [
            SearchResult(
                book_title=row[0],
                author=row[1],
                chapter_title=row[2],
                chapter_number=row[3],
                page_number=row[4],
                chunk_index=row[5],
                content=row[6],
                score=float(row[7]),
            )
            for row in rows
        ]

    def search_vector(
        self, *, query: str, limit: int, book_filter: str | None = None
    ) -> list[SearchResult]:
        vector = self.embeddings.embed_query(query)
        rows = self.db.search_vector(
            embedding=vector,
            limit=limit,
            book_filter=book_filter,
        )
        return self._rows_to_results(rows)

    def search_keyword(
        self, *, query: str, limit: int, book_filter: str | None = None
    ) -> list[SearchResult]:
        rows = self.db.search_keyword(
            query_text=query,
            limit=limit,
            book_filter=book_filter,
        )
        return self._rows_to_results(rows)
