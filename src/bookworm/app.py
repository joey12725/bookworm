"""The :class:`Bookworm` facade: one object that wires settings, storage, embeddings,
retrieval and Q&A together. The CLI and the MCP server are thin layers over it, and it is
the API to use from Python::

    from bookworm import Bookworm, Settings

    with Bookworm.open(Settings(db_path="library.duckdb")) as bw:
        bw.index(["~/Books"])
        for hit in bw.search("first impressions"):
            print(hit.citation.format(), hit.text[:80])
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from bookworm.chunking import ChunkingConfig
from bookworm.config import Settings
from bookworm.embeddings import Embedder, create_embedder
from bookworm.errors import BookNotFoundError, BookwormError
from bookworm.goodreads import GoodreadsReport, import_goodreads
from bookworm.indexer import Indexer, ProgressCallback
from bookworm.llm import ChatModel, create_chat_model
from bookworm.models import Answer, Book, Hit, IndexReport, SearchMode
from bookworm.qa import answer
from bookworm.search import AdjacentDedupe, MMRReranker, Reranker, Retriever
from bookworm.store import Store
from bookworm.text import fold


class Bookworm:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        chat: ChatModel | None = None,
        store: Store | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or Store(settings.db_path)
        self._embedder = embedder
        self._chat = chat
        self._chat_resolved = chat is not None

    @classmethod
    def open(cls, settings: Settings | None = None, **kwargs: Any) -> Bookworm:
        return cls(settings or Settings.from_env(), **kwargs)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Bookworm:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Providers are created lazily so read-only commands never need API keys.
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = create_embedder(self.settings)
        return self._embedder

    @property
    def chat(self) -> ChatModel | None:
        if not self._chat_resolved:
            self._chat = create_chat_model(self.settings)
            self._chat_resolved = True
        return self._chat

    @property
    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(
            target_tokens=self.settings.chunk_tokens, overlap_tokens=self.settings.chunk_overlap
        )

    def _indexer(self) -> Indexer:
        return Indexer(self.store, self.embedder, self.chunking)

    def _retriever(self) -> Retriever:
        rerankers: list[Reranker] = [AdjacentDedupe()]
        if self.settings.mmr_lambda < 1:
            rerankers.append(MMRReranker(self.settings.mmr_lambda))
        return Retriever(self.store, self.embedder, rrf_k=self.settings.rrf_k, rerankers=rerankers)

    # -- indexing ---------------------------------------------------------------------

    def index(
        self,
        paths: Iterable[Path | str] | None = None,
        *,
        dry_run: bool = False,
        force: bool = False,
        prune: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IndexReport:
        roots = [Path(p) for p in paths] if paths else []
        if not roots:
            if self.settings.library_path is None:
                raise BookwormError("No path given and BOOKWORM_LIBRARY is not set.")
            roots = [self.settings.library_path]
        return self._indexer().index(
            roots,
            dry_run=dry_run,
            force=force,
            prune=prune,
            excluded_dirs=self.settings.excluded_dirs,
            progress=progress,
        )

    def reindex(self, progress: Any = None) -> int:
        return self._indexer().reindex(progress=progress)

    def remove(self, title: str) -> Book:
        book = self.find_book(title)
        self.store.delete_book(book.id)
        return book

    # -- lookup -----------------------------------------------------------------------

    def books(self) -> list[Book]:
        return self.store.list_books()

    def match_books(self, query: str) -> list[Book]:
        """Books whose title contains ``query``, or the closest fuzzy match."""
        books = self.store.list_books()
        q = fold(query.strip())
        exact = [b for b in books if fold(b.title) == q]
        if exact:
            return exact
        contains = [b for b in books if q in fold(b.title)]
        if contains:
            return contains
        scored = [(SequenceMatcher(None, q, fold(b.title)).ratio(), b) for b in books]
        best = max(scored, key=lambda s: s[0], default=None)
        return [best[1]] if best and best[0] >= 0.6 else []

    def find_book(self, query: str) -> Book:
        matches = self.match_books(query)
        if not matches:
            raise BookNotFoundError(f"No book matching {query!r}")
        return matches[0]

    def chapters(self, title: str) -> tuple[Book, list[tuple[int, str | None, int]]]:
        book = self.find_book(title)
        return book, self.store.chapters(book.id)

    def read(self, title: str, chapter: int) -> tuple[Book, str | None, str]:
        book = self.find_book(title)
        found = self.store.chapter_text(book.id, chapter)
        if found is None:
            raise BookNotFoundError(f"{book.title} has no chapter {chapter}")
        chapter_title, text = found
        return book, chapter_title, text

    def stats(self) -> dict[str, Any]:
        stats = self.store.stats()
        if self.store.path.exists():
            stats["db_bytes"] = self.store.path.stat().st_size
        return stats

    # -- retrieval --------------------------------------------------------------------

    def _book_ids(self, book: str | None) -> Sequence[int] | None:
        if book is None:
            return None
        matches = self.match_books(book)
        if not matches:
            raise BookNotFoundError(f"No book matching {book!r}")
        return [b.id for b in matches]

    def search(
        self, query: str, *, limit: int = 5, mode: SearchMode = "hybrid", book: str | None = None
    ) -> list[Hit]:
        return self._retriever().search(
            query, limit=limit, mode=mode, book_ids=self._book_ids(book)
        )

    def ask(self, question: str, *, limit: int = 6, book: str | None = None) -> Answer:
        hits = self.search(question, limit=limit, book=book)
        return answer(question, hits, self.chat)

    # -- extras -----------------------------------------------------------------------

    def import_goodreads(self, csv_path: Path) -> GoodreadsReport:
        return import_goodreads(self.store, csv_path)

    def backup(self, destination: Path) -> Path:
        self.store.checkpoint()
        destination = destination.expanduser()
        if destination.is_dir():
            destination = destination / self.store.path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.store.path, destination)
        return destination


def restore(source: Path, db_path: Path) -> Path:
    """Replace the database file with a backup. Run with no open connection."""
    if not source.is_file():
        raise BookwormError(f"Backup not found: {source}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, db_path)
    wal = db_path.with_name(db_path.name + ".wal")
    wal.unlink(missing_ok=True)
    return db_path
