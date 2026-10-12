from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import duckdb

from bookworm_core.config import EMBEDDING_DIMENSIONS
from bookworm_core.models import ChunkRecord


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self.conn.execute("PRAGMA enable_progress_bar=false;")
        self._similarity_function: str | None = None
        self._initialize()

    def _initialize(self) -> None:
        self._load_extension("vss")
        self._initialize_schema()
        self._maybe_create_vector_index()

    def _load_extension(self, extension: str) -> None:
        try:
            self.conn.execute(f"INSTALL {extension};")
        except duckdb.Error:
            # Extension may already be installed or unavailable in constrained environments.
            pass

        try:
            self.conn.execute(f"LOAD {extension};")
        except duckdb.Error:
            # Keep working even if vector indexing is unavailable; vector columns still function.
            pass

    def _initialize_schema(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS books (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                file_path TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                format TEXT NOT NULL,
                file_size_bytes BIGINT NOT NULL,
                total_pages INTEGER,
                goodreads_rating FLOAT,
                goodreads_read_status TEXT,
                goodreads_shelves TEXT,
                added_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL,
                chapter_title TEXT,
                chapter_number INTEGER,
                page_number INTEGER,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                embedding FLOAT[{EMBEDDING_DIMENSIONS}],
                FOREIGN KEY (book_id) REFERENCES books(id)
            );

            CREATE INDEX IF NOT EXISTS idx_books_hash ON books(file_hash);
            CREATE INDEX IF NOT EXISTS idx_books_title ON books(title);
            CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id);
            """
        )

    def _maybe_create_vector_index(self) -> None:
        try:
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                ON chunks USING HNSW (embedding)
                WITH (metric = 'cosine');
                """
            )
        except duckdb.Error:
            # Keep query-time fallback if extension syntax is unavailable.
            pass

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN TRANSACTION;")
        try:
            yield
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise
        else:
            self.conn.execute("COMMIT;")

    def get_book_by_path(self, file_path: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM books WHERE file_path = ? LIMIT 1;", [file_path]
        ).fetchone()
        return self._row_to_dict("books", row) if row else None

    def list_books(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM books ORDER BY title ASC;").fetchall()
        return [self._row_to_dict("books", r) for r in rows]

    def get_book_by_id(self, book_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM books WHERE id = ?;", [book_id]).fetchone()
        return self._row_to_dict("books", row) if row else None

    def upsert_book(
        self,
        *,
        title: str,
        author: str | None,
        file_path: str,
        file_hash: str,
        file_format: str,
        file_size_bytes: int,
        total_pages: int | None,
        existing_id: str | None = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        if existing_id:
            self.conn.execute(
                """
                UPDATE books
                SET title = ?,
                    author = ?,
                    file_hash = ?,
                    format = ?,
                    file_size_bytes = ?,
                    total_pages = ?,
                    updated_at = ?
                WHERE id = ?;
                """,
                [
                    title,
                    author,
                    file_hash,
                    file_format,
                    file_size_bytes,
                    total_pages,
                    now,
                    existing_id,
                ],
            )
            return existing_id

        book_id = str(uuid4())
        self.conn.execute(
            """
            INSERT INTO books (
                id, title, author, file_path, file_hash, format,
                file_size_bytes, total_pages,
                goodreads_rating, goodreads_read_status, goodreads_shelves,
                added_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?);
            """,
            [
                book_id,
                title,
                author,
                file_path,
                file_hash,
                file_format,
                file_size_bytes,
                total_pages,
                now,
                now,
            ],
        )
        return book_id

    def delete_chunks_for_book(self, book_id: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE book_id = ?;", [book_id])

    def delete_book(self, book_id: str) -> None:
        with self.transaction():
            self.delete_chunks_for_book(book_id)
            self.conn.execute("DELETE FROM books WHERE id = ?;", [book_id])

    def insert_chunks(
        self,
        book_id: str,
        chunks: Iterable[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            rows.append(
                [
                    str(uuid4()),
                    book_id,
                    chunk.chapter_title,
                    chunk.chapter_number,
                    chunk.page_number,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.token_count,
                    embedding,
                ]
            )

        self.conn.executemany(
            f"""
            INSERT INTO chunks (
                id, book_id, chapter_title, chapter_number, page_number,
                chunk_index, content, token_count, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::FLOAT[{EMBEDDING_DIMENSIONS}]);
            """,
            rows,
        )

    def _detect_similarity_function(self) -> str:
        if self._similarity_function:
            return self._similarity_function

        candidates = [
            "array_cosine_similarity",
            "list_cosine_similarity",
            "array_inner_product",
            "list_inner_product",
        ]

        existing = {
            name
            for (name,) in self.conn.execute(
                "SELECT DISTINCT function_name FROM duckdb_functions();"
            ).fetchall()
        }

        for fn in candidates:
            if fn in existing:
                self._similarity_function = fn
                return fn

        raise RuntimeError(
            "No vector similarity function found in DuckDB. Ensure your DuckDB build supports arrays/lists and VSS."
        )

    def search_vector(
        self,
        *,
        embedding: list[float],
        limit: int,
        book_filter: str | None,
    ) -> list[tuple[Any, ...]]:
        similarity_fn = self._detect_similarity_function()
        similarity_expr = (
            f"{similarity_fn}(c.embedding, ?::FLOAT[{EMBEDDING_DIMENSIONS}])"
        )

        params: list[Any] = [embedding]
        where_clauses = []

        if book_filter:
            where_clauses.append("lower(b.title) LIKE '%' || lower(?) || '%'")
            params.append(book_filter)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
            SELECT
                b.title,
                b.author,
                c.chapter_title,
                c.chapter_number,
                c.page_number,
                c.chunk_index,
                c.content,
                {similarity_expr} AS score
            FROM chunks c
            JOIN books b ON b.id = c.book_id
            {where_sql}
            ORDER BY score DESC
            LIMIT ?;
        """

        params.append(limit)
        return self.conn.execute(query, params).fetchall()

    def search_keyword(
        self,
        *,
        query_text: str,
        limit: int,
        book_filter: str | None,
    ) -> list[tuple[Any, ...]]:
        params: list[Any] = [query_text, query_text, query_text]
        where_clauses = ["lower(c.content) LIKE '%' || lower(?) || '%'"]

        if book_filter:
            where_clauses.append("lower(b.title) LIKE '%' || lower(?) || '%'")
            params.append(book_filter)

        where_sql = " AND ".join(where_clauses)
        params.append(limit)

        return self.conn.execute(
            f"""
            SELECT
                b.title,
                b.author,
                c.chapter_title,
                c.chapter_number,
                c.page_number,
                c.chunk_index,
                c.content,
                (
                    (length(lower(c.content)) - length(replace(lower(c.content), lower(?), '')))
                    / NULLIF(length(?), 0)
                )::DOUBLE AS score
            FROM chunks c
            JOIN books b ON b.id = c.book_id
            WHERE {where_sql}
            ORDER BY score DESC, b.title ASC, c.chunk_index ASC
            LIMIT ?;
            """,
            params,
        ).fetchall()

    def get_book_chunks_for_chapter(
        self, book_id: str, chapter_number: int
    ) -> list[tuple[Any, ...]]:
        return self.conn.execute(
            """
            SELECT chapter_title, chunk_index, content
            FROM chunks
            WHERE book_id = ? AND chapter_number = ?
            ORDER BY chunk_index ASC;
            """,
            [book_id, chapter_number],
        ).fetchall()

    def get_stats(self) -> dict[str, Any]:
        totals = self.conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM books) AS total_books,
                (SELECT COUNT(*) FROM chunks) AS total_chunks,
                (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL) AS total_embeddings;
            """
        ).fetchone()

        by_format = self.conn.execute(
            "SELECT format, COUNT(*) FROM books GROUP BY format ORDER BY COUNT(*) DESC;"
        ).fetchall()
        top_authors = self.conn.execute(
            """
            SELECT COALESCE(author, 'Unknown') AS author, COUNT(*)
            FROM books
            GROUP BY author
            ORDER BY COUNT(*) DESC, author ASC
            LIMIT 10;
            """
        ).fetchall()
        recent = self.conn.execute(
            """
            SELECT title, author, added_at
            FROM books
            ORDER BY added_at DESC
            LIMIT 10;
            """
        ).fetchall()

        return {
            "total_books": totals[0],
            "total_chunks": totals[1],
            "total_embeddings": totals[2],
            "by_format": by_format,
            "top_authors": top_authors,
            "recent": recent,
        }

    def update_goodreads_metadata(
        self,
        *,
        book_id: str,
        rating: float | None,
        read_status: str | None,
        shelves: str | None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE books
            SET goodreads_rating = ?,
                goodreads_read_status = ?,
                goodreads_shelves = ?,
                updated_at = ?
            WHERE id = ?;
            """,
            [rating, read_status, shelves, datetime.now(timezone.utc), book_id],
        )

    def _row_to_dict(self, table: str, row: tuple[Any, ...]) -> dict[str, Any]:
        if table == "books":
            keys = [
                "id",
                "title",
                "author",
                "file_path",
                "file_hash",
                "format",
                "file_size_bytes",
                "total_pages",
                "goodreads_rating",
                "goodreads_read_status",
                "goodreads_shelves",
                "added_at",
                "updated_at",
            ]
            return dict(zip(keys, row, strict=True))

        raise ValueError(f"Unsupported table mapping: {table}")
