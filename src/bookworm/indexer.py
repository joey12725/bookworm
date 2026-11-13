"""Incremental indexing.

For each file the indexer takes the cheapest path that is still correct:

1. same path, same size and mtime: skip without reading the file;
2. same content hash as a book already indexed (touched, or moved/renamed): update the
   record, no re-extraction;
3. changed content: re-extract and re-chunk, but reuse the stored embedding of every chunk
   whose text is unchanged, so editing one chapter re-embeds one chapter.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from bookworm.chunking import ChunkingConfig, chunk_sections
from bookworm.embeddings import Embedder
from bookworm.errors import BookwormError
from bookworm.extractors import extract, supported_extensions
from bookworm.models import FileResult, FileStatus, IndexReport
from bookworm.store import Store
from bookworm.text import sha256_text

ProgressCallback = Callable[[FileResult, int, int], None]


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def discover(paths: Iterable[Path], excluded_dirs: Iterable[str] = ()) -> list[Path]:
    """Expand files and directories into a sorted list of supported book files."""
    excluded = {d.lower() for d in excluded_dirs}
    extensions = supported_extensions()
    found: set[Path] = set()
    for root in paths:
        root = root.expanduser().resolve()
        if root.is_file():
            found.add(root)
            continue
        if not root.is_dir():
            raise BookwormError(f"Path does not exist: {root}")
        for path in root.rglob("*"):
            rel_parts = path.relative_to(root).parts
            if any(part.startswith(".") or part.lower() in excluded for part in rel_parts):
                continue
            if path.is_file() and path.suffix.lower() in extensions:
                found.add(path)
    return sorted(found)


@dataclass(slots=True)
class Indexer:
    store: Store
    embedder: Embedder
    chunking: ChunkingConfig

    def index(
        self,
        paths: Iterable[Path],
        *,
        dry_run: bool = False,
        force: bool = False,
        prune: bool = False,
        excluded_dirs: Iterable[str] = (),
        progress: ProgressCallback | None = None,
    ) -> IndexReport:
        roots = list(paths)
        files = discover(roots, excluded_dirs)
        # Chunking settings changed since the last run: every book needs re-chunking.
        if self.store.get_meta("chunking") not in (None, self.chunking.signature):
            force = True
        if not dry_run:
            self.store.bind_embedder(self.embedder.name)
            self.store.set_meta("chunking", self.chunking.signature)
        report = IndexReport()
        for n, path in enumerate(files, start=1):
            try:
                result = self.index_file(path, dry_run=dry_run, force=force)
            except Exception as exc:  # report and keep going; one bad file is not fatal
                result = FileResult(str(path), "failed", error=f"{type(exc).__name__}: {exc}")
            report.files.append(result)
            if progress:
                progress(result, n, len(files))
        if prune:
            report.removed = self.prune(roots, dry_run=dry_run)
        if not dry_run:
            self.store.checkpoint()
        return report

    def index_file(self, path: Path, *, dry_run: bool = False, force: bool = False) -> FileResult:
        path = path.expanduser().resolve()
        key = str(path)
        stat = path.stat()
        known = self.store.book_by_path(key)

        if known and not force:
            book, mtime = known
            if book.size_bytes == stat.st_size and mtime == stat.st_mtime:
                return FileResult(key, "unchanged", title=book.title, chunks=book.chunks)

        digest = sha256_file(path)
        if known and not force and known[0].sha256 == digest:
            if not dry_run:
                self.store.touch_book(known[0].id, stat.st_mtime)
            return FileResult(key, "unchanged", title=known[0].title, chunks=known[0].chunks)

        if not known and not force:
            moved = [b for b in self.store.books_by_hash(digest) if not Path(b.path).exists()]
            if moved:
                book = moved[0]
                if not dry_run:
                    self.store.update_book(book.id, path=key, mtime=stat.st_mtime)
                return FileResult(key, "moved", title=book.title, chunks=book.chunks)

        document = extract(path)
        chunks = chunk_sections(document.sections, self.chunking)
        if not chunks:
            raise BookwormError("No text could be extracted")
        if dry_run:
            status: FileStatus = "would-update" if known else "would-add"
            return FileResult(key, status, title=document.title, chunks=len(chunks))

        hashes = [sha256_text(c.text) for c in chunks]
        cached = self.store.cached_vectors(hashes)
        missing = [i for i, h in enumerate(hashes) if h not in cached]
        fresh = self.embedder.embed_documents([chunks[i].text for i in missing])
        new_vectors = dict(zip(missing, fresh, strict=True))
        vectors = [cached[h] if h in cached else new_vectors[i] for i, h in enumerate(hashes)]

        with self.store.transaction():
            if known:
                book_id = known[0].id
                self.store.update_book(
                    book_id, title=document.title, author=document.author,
                    format=document.format, sha256=digest, size_bytes=stat.st_size,
                    mtime=stat.st_mtime, pages=document.pages, chapters=document.chapter_count,
                )  # fmt: skip
            else:
                book_id = self.store.insert_book(
                    path=key, title=document.title, author=document.author,
                    format=document.format, sha256=digest, size_bytes=stat.st_size,
                    mtime=stat.st_mtime, pages=document.pages, chapters=document.chapter_count,
                )  # fmt: skip
            self.store.replace_chunks(book_id, chunks, hashes, vectors)

        return FileResult(
            key,
            "updated" if known else "added",
            title=document.title,
            chunks=len(chunks),
            reused_vectors=len(chunks) - len(missing),
        )

    def prune(self, roots: Iterable[Path], *, dry_run: bool = False) -> list[str]:
        """Remove books under ``roots`` whose files no longer exist."""
        resolved = [r.expanduser().resolve() for r in roots]
        removed: list[str] = []
        for book in self.store.list_books():
            path = Path(book.path)
            under_root = any(path == r or r in path.parents for r in resolved)
            if under_root and not path.exists():
                removed.append(book.path)
                if not dry_run:
                    self.store.delete_book(book.id)
        return removed

    def reindex(self, *, progress: Callable[[int, int], None] | None = None) -> int:
        """Re-embed every chunk with the current embedder (after switching models)."""
        rows = self.store.all_chunk_texts()
        batch = 128
        with self.store.transaction():
            for start in range(0, len(rows), batch):
                part = rows[start : start + batch]
                vectors = self.embedder.embed_documents([text for _, _, text in part])
                self.store.set_embeddings([cid for cid, _, _ in part], vectors)
                if progress:
                    progress(min(start + batch, len(rows)), len(rows))
            self.store.set_meta("embedder", self.embedder.name)
        self.store.checkpoint()
        return len(rows)
