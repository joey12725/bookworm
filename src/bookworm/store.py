"""DuckDB storage: books, chunks with embeddings, and a BM25 inverted index, in one file.

Vector search is an exact cosine scan with ``list_dot_product`` over L2-normalised vectors,
which DuckDB runs fast enough for libraries of hundreds of books with no extension to
install. BM25 is computed in SQL over a ``postings`` table built at index time.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from bookworm.errors import IndexMismatchError
from bookworm.models import Book, Chunk, Citation, Hit
from bookworm.text import terms

SCHEMA_VERSION = "2"

_SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS book_ids START 1;
CREATE SEQUENCE IF NOT EXISTS chunk_ids START 1;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT,
    value TEXT
);

CREATE TABLE IF NOT EXISTS books (
    id BIGINT,
    path TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    format TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    mtime DOUBLE NOT NULL,
    pages INTEGER,
    chapters INTEGER NOT NULL,
    added_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    rating DOUBLE,
    shelf TEXT,
    shelves TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGINT,
    book_id BIGINT NOT NULL,
    ord INTEGER NOT NULL,
    chapter INTEGER,
    chapter_title TEXT,
    section TEXT,
    page_start INTEGER,
    page_end INTEGER,
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    overlap_chars INTEGER NOT NULL,
    n_terms INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    embedding FLOAT[]
);

CREATE TABLE IF NOT EXISTS postings (
    term TEXT NOT NULL,
    chunk_id BIGINT NOT NULL,
    tf INTEGER NOT NULL
);
"""

_BOOK_COLUMNS = (
    "b.id, b.title, b.author, b.path, b.format, b.sha256, b.size_bytes, b.pages, b.chapters, "
    "(SELECT count(*) FROM chunks c WHERE c.book_id = b.id) AS chunks, "
    "b.added_at, b.updated_at, b.rating, b.shelf, b.shelves"
)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)  # stored as naive UTC


class Store:
    def __init__(self, path: Path | str, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if not read_only and str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(path), read_only=read_only)
        if not read_only:
            self.conn.execute(_SCHEMA)
            if self.get_meta("schema_version") is None:
                self.set_meta("schema_version", SCHEMA_VERSION)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.conn.execute("BEGIN TRANSACTION")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    # -- metadata ---------------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("DELETE FROM meta WHERE key = ?", [key])
        self.conn.execute("INSERT INTO meta VALUES (?, ?)", [key, value])

    def chunk_count(self) -> int:
        row = self.conn.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def bind_embedder(self, name: str) -> None:
        """Record the embedder for a fresh index, or refuse a mismatched one."""
        current = self.get_meta("embedder")
        if current is None or self.chunk_count() == 0:
            if not self.read_only:
                self.set_meta("embedder", name)
        elif current != name:
            raise IndexMismatchError(
                f"This library was indexed with embedder {current!r} but {name!r} is configured. "
                "Run `bookworm reindex` to re-embed everything with the new model."
            )

    # -- books ------------------------------------------------------------------------

    def _book(self, row: Sequence[Any]) -> Book:
        return Book(
            id=row[0], title=row[1], author=row[2], path=row[3], format=row[4], sha256=row[5],
            size_bytes=row[6], pages=row[7], chapters=row[8], chunks=row[9], added_at=row[10],
            updated_at=row[11], rating=row[12], shelf=row[13], shelves=row[14],
        )  # fmt: skip

    def list_books(self) -> list[Book]:
        rows = self.conn.execute(
            f"SELECT {_BOOK_COLUMNS} FROM books b ORDER BY lower(b.title)"
        ).fetchall()
        return [self._book(r) for r in rows]

    def get_book(self, book_id: int) -> Book | None:
        row = self.conn.execute(
            f"SELECT {_BOOK_COLUMNS} FROM books b WHERE b.id = ?", [book_id]
        ).fetchone()
        return self._book(row) if row else None

    def book_by_path(self, path: str) -> tuple[Book, float] | None:
        row = self.conn.execute(
            f"SELECT {_BOOK_COLUMNS}, b.mtime FROM books b WHERE b.path = ?", [path]
        ).fetchone()
        return (self._book(row), float(row[15])) if row else None

    def books_by_hash(self, sha256: str) -> list[Book]:
        rows = self.conn.execute(
            f"SELECT {_BOOK_COLUMNS} FROM books b WHERE b.sha256 = ?", [sha256]
        ).fetchall()
        return [self._book(r) for r in rows]

    def insert_book(
        self, *, path: str, title: str, author: str | None, format: str, sha256: str,
        size_bytes: int, mtime: float, pages: int | None, chapters: int,
    ) -> int:  # fmt: skip
        now = _now()
        row = self.conn.execute(
            """
            INSERT INTO books (id, path, title, author, format, sha256, size_bytes, mtime,
                               pages, chapters, added_at, updated_at)
            VALUES (nextval('book_ids'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            [path, title, author, format, sha256, size_bytes, mtime, pages, chapters, now, now],
        ).fetchone()
        assert row is not None
        return int(row[0])

    def update_book(self, book_id: int, **fields: Any) -> None:
        allowed = {
            "path", "title", "author", "format", "sha256", "size_bytes", "mtime", "pages",
            "chapters", "rating", "shelf", "shelves",
        }  # fmt: skip
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown book fields: {sorted(unknown)}")
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        self.conn.execute(
            f"UPDATE books SET {assignments} WHERE id = ?", [*fields.values(), book_id]
        )

    def touch_book(self, book_id: int, mtime: float) -> None:
        """Record a new mtime without bumping updated_at (content did not change)."""
        self.conn.execute("UPDATE books SET mtime = ? WHERE id = ?", [mtime, book_id])

    def delete_book(self, book_id: int) -> None:
        with self.transaction():
            self._delete_chunks(book_id)
            self.conn.execute("DELETE FROM books WHERE id = ?", [book_id])

    # -- chunks -----------------------------------------------------------------------

    def _delete_chunks(self, book_id: int) -> None:
        self.conn.execute(
            "DELETE FROM postings WHERE chunk_id IN (SELECT id FROM chunks WHERE book_id = ?)",
            [book_id],
        )
        self.conn.execute("DELETE FROM chunks WHERE book_id = ?", [book_id])

    def cached_vectors(self, hashes: Sequence[str]) -> dict[str, list[float]]:
        """Embeddings already computed for identical chunk text anywhere in the library."""
        if not hashes:
            return {}
        rows = self.conn.execute(
            """
            SELECT content_hash, any_value(embedding)
            FROM chunks
            WHERE embedding IS NOT NULL AND content_hash IN (SELECT unnest(?::VARCHAR[]))
            GROUP BY content_hash
            """,
            [list(set(hashes))],
        ).fetchall()
        return {h: list(v) for h, v in rows}

    def replace_chunks(
        self,
        book_id: int,
        chunks: Sequence[Chunk],
        hashes: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Swap a book's chunks and postings. Call inside :meth:`transaction`."""
        self._delete_chunks(book_id)
        if not chunks:
            return
        id_row = self.conn.execute(
            "SELECT list(nextval('chunk_ids')) FROM range(?)", [len(chunks)]
        ).fetchone()
        assert id_row is not None
        ids: list[int] = list(id_row[0])

        chunk_terms = [Counter(terms(c.text)) for c in chunks]
        columns: dict[str, list[Any]] = {
            "id": ids,
            "ord": [c.ord for c in chunks],
            "chapter": [c.chapter for c in chunks],
            "chapter_title": [c.chapter_title for c in chunks],
            "section": [c.section for c in chunks],
            "page_start": [c.page_start for c in chunks],
            "page_end": [c.page_end for c in chunks],
            "text": [c.text for c in chunks],
            "tokens": [c.tokens for c in chunks],
            "overlap_chars": [c.overlap_chars for c in chunks],
            "n_terms": [sum(t.values()) for t in chunk_terms],
            "content_hash": list(hashes),
            "embedding": [list(v) for v in vectors],
        }
        types = {
            "id": "BIGINT[]", "ord": "INTEGER[]", "chapter": "INTEGER[]",
            "chapter_title": "VARCHAR[]", "section": "VARCHAR[]", "page_start": "INTEGER[]",
            "page_end": "INTEGER[]", "text": "VARCHAR[]", "tokens": "INTEGER[]",
            "overlap_chars": "INTEGER[]", "n_terms": "INTEGER[]", "content_hash": "VARCHAR[]",
            "embedding": "FLOAT[][]",
        }  # fmt: skip
        select = ", ".join(f"unnest(?::{types[name]})" for name in columns)
        self.conn.execute(
            f"INSERT INTO chunks (book_id, {', '.join(columns)}) SELECT ?, {select}",
            [book_id, *columns.values()],
        )

        post_terms: list[str] = []
        post_ids: list[int] = []
        post_tf: list[int] = []
        for chunk_id, counter in zip(ids, chunk_terms, strict=True):
            for term, tf in counter.items():
                post_terms.append(term)
                post_ids.append(chunk_id)
                post_tf.append(tf)
        if post_terms:
            self.conn.execute(
                "INSERT INTO postings SELECT unnest(?::VARCHAR[]), unnest(?::BIGINT[]), "
                "unnest(?::INTEGER[])",
                [post_terms, post_ids, post_tf],
            )

    def all_chunk_texts(self) -> list[tuple[int, str, str]]:
        rows = self.conn.execute("SELECT id, content_hash, text FROM chunks ORDER BY id").fetchall()
        return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]

    def set_embeddings(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        self.conn.execute(
            """
            UPDATE chunks SET embedding = v.embedding
            FROM (SELECT unnest(?::BIGINT[]) AS id, unnest(?::FLOAT[][]) AS embedding) v
            WHERE chunks.id = v.id
            """,
            [list(ids), [list(v) for v in vectors]],
        )

    # -- retrieval --------------------------------------------------------------------

    @staticmethod
    def _filters(
        book_ids: Sequence[int] | None, phrases: Sequence[str] = ()
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if book_ids is not None:
            clauses.append("c.book_id IN (SELECT unnest(?::BIGINT[]))")
            params.append(list(book_ids))
        for phrase in phrases:
            clauses.append("strpos(lower(c.text), lower(?)) > 0")
            params.append(phrase)
        return (" AND " + " AND ".join(clauses) if clauses else ""), params

    def search_bm25(
        self,
        query_terms: Sequence[str],
        *,
        limit: int,
        book_ids: Sequence[int] | None = None,
        phrases: Sequence[str] = (),
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[tuple[int, float]]:
        where, params = self._filters(book_ids, phrases)
        unique_terms = list(dict.fromkeys(query_terms))
        if not unique_terms:
            if not phrases:
                return []
            # Phrase-only query: rank matching chunks by how early they appear.
            rows = self.conn.execute(
                f"SELECT c.id, 1.0 FROM chunks c WHERE TRUE {where} ORDER BY c.book_id, c.ord "
                "LIMIT ?",
                [*params, limit],
            ).fetchall()
            return [(int(r[0]), float(r[1])) for r in rows]
        rows = self.conn.execute(
            f"""
            WITH q AS (SELECT DISTINCT unnest(?::VARCHAR[]) AS term),
            stats AS (
                SELECT count(*)::DOUBLE AS n, coalesce(avg(n_terms), 1)::DOUBLE AS avgdl
                FROM chunks
            ),
            df AS (
                SELECT p.term, count(*)::DOUBLE AS df
                FROM postings p JOIN q ON p.term = q.term
                GROUP BY p.term
            )
            SELECT c.id,
                   sum(
                       ln(1 + (stats.n - df.df + 0.5) / (df.df + 0.5))
                       * (p.tf * (? + 1))
                       / (p.tf + ? * (1 - ? + ? * c.n_terms / stats.avgdl))
                   ) AS score
            FROM postings p
            JOIN df ON p.term = df.term
            JOIN chunks c ON c.id = p.chunk_id
            CROSS JOIN stats
            WHERE TRUE {where}
            GROUP BY c.id
            ORDER BY score DESC, c.id
            LIMIT ?
            """,
            [unique_terms, k1, k1, b, b, *params, limit],
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def search_vector(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        book_ids: Sequence[int] | None = None,
        phrases: Sequence[str] = (),
    ) -> list[tuple[int, float]]:
        where, params = self._filters(book_ids, phrases)
        rows = self.conn.execute(
            f"""
            SELECT c.id, list_dot_product(c.embedding, ?::FLOAT[]) AS score
            FROM chunks c
            WHERE c.embedding IS NOT NULL {where}
            ORDER BY score DESC, c.id
            LIMIT ?
            """,
            [list(vector), *params, limit],
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def embeddings_for(self, chunk_ids: Sequence[int]) -> dict[int, list[float]]:
        rows = self.conn.execute(
            "SELECT id, embedding FROM chunks WHERE id IN (SELECT unnest(?::BIGINT[]))",
            [list(chunk_ids)],
        ).fetchall()
        return {int(r[0]): list(r[1]) for r in rows if r[1] is not None}

    def hits(self, scored: Sequence[tuple[int, float]]) -> list[Hit]:
        """Hydrate (chunk_id, score) pairs into Hits, preserving order."""
        if not scored:
            return []
        rows = self.conn.execute(
            """
            SELECT c.id, c.book_id, c.ord, c.text, b.title, b.author, c.chapter,
                   c.chapter_title, c.section, c.page_start, c.page_end
            FROM chunks c JOIN books b ON b.id = c.book_id
            WHERE c.id IN (SELECT unnest(?::BIGINT[]))
            """,
            [[cid for cid, _ in scored]],
        ).fetchall()
        by_id = {int(r[0]): r for r in rows}
        hits: list[Hit] = []
        for chunk_id, score in scored:
            r = by_id.get(chunk_id)
            if r is None:
                continue
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    book_id=int(r[1]),
                    ord=int(r[2]),
                    text=str(r[3]),
                    score=score,
                    citation=Citation(
                        book=r[4],
                        author=r[5],
                        chapter=r[6],
                        chapter_title=r[7],
                        section=r[8],
                        page_start=r[9],
                        page_end=r[10],
                    ),
                )
            )
        return hits

    # -- reading ----------------------------------------------------------------------

    def chapters(self, book_id: int) -> list[tuple[int, str | None, int]]:
        """(chapter number, title, chunk count) in reading order."""
        rows = self.conn.execute(
            """
            SELECT chapter, any_value(chapter_title), count(*)
            FROM chunks WHERE book_id = ? AND chapter IS NOT NULL
            GROUP BY chapter ORDER BY chapter
            """,
            [book_id],
        ).fetchall()
        return [(int(r[0]), r[1], int(r[2])) for r in rows]

    def chapter_text(self, book_id: int, chapter: int | None) -> tuple[str | None, str] | None:
        """Reassemble a chapter's text from its chunks, dropping the overlap."""
        rows = self.conn.execute(
            """
            SELECT chapter_title, section, text, overlap_chars FROM chunks
            WHERE book_id = ? AND chapter IS NOT DISTINCT FROM ?
            ORDER BY ord
            """,
            [book_id, chapter],
        ).fetchall()
        if not rows:
            return None
        parts: list[str] = []
        last_section: str | None = None
        for _, section, text, overlap in rows:
            if section and section != last_section:
                parts.append(f"## {section}")
            last_section = section
            parts.append(text[overlap:] if overlap else text)
        return rows[0][0], "\n\n".join(parts)

    def neighbours(self, book_id: int, ord_: int, radius: int = 1) -> list[tuple[int, str, int]]:
        rows = self.conn.execute(
            """
            SELECT ord, text, overlap_chars FROM chunks
            WHERE book_id = ? AND ord BETWEEN ? AND ? ORDER BY ord
            """,
            [book_id, ord_ - radius, ord_ + radius],
        ).fetchall()
        return [(int(r[0]), str(r[1]), int(r[2])) for r in rows]

    def stats(self) -> dict[str, Any]:
        row = self.conn.execute(
            """
            SELECT (SELECT count(*) FROM books), (SELECT count(*) FROM chunks),
                   (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL),
                   (SELECT coalesce(sum(tokens), 0) FROM chunks),
                   (SELECT count(DISTINCT term) FROM postings)
            """
        ).fetchone()
        assert row is not None
        return {
            "books": row[0],
            "chunks": row[1],
            "embedded": row[2],
            "tokens": row[3],
            "vocabulary": row[4],
            "embedder": self.get_meta("embedder") or "-",
            "by_format": self.conn.execute(
                "SELECT format, count(*) FROM books GROUP BY format ORDER BY count(*) DESC, format"
            ).fetchall(),
            "top_authors": self.conn.execute(
                """
                SELECT coalesce(author, 'Unknown'), count(*) FROM books
                GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 10
                """
            ).fetchall(),
        }

    def checkpoint(self) -> None:
        self.conn.execute("CHECKPOINT")
