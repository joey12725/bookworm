from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bookworm_core.chunking import chunk_fragments
from bookworm_core.db import Database
from bookworm_core.embeddings import EmbeddingService
from bookworm_core.errors import IngestionError
from bookworm_core.extractors import extract_book
from bookworm_core.models import ScanResult
from bookworm_core.utils import SUPPORTED_EXTENSIONS, sha256_file


@dataclass(slots=True)
class IngestOutcome:
    status: str
    chunks: int = 0
    title: str | None = None
    reason: str | None = None


EXCLUDED_DIRS = {"manga"}


def discover_ebook_files(root: Path) -> list[Path]:
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part.lower() in EXCLUDED_DIRS for part in p.parts)
    ]
    return sorted(files)


class IngestionService:
    def __init__(self, db: Database, embeddings: EmbeddingService) -> None:
        self.db = db
        self.embeddings = embeddings

    def ingest_file(
        self,
        path: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
        embed_progress_callback: Callable[[int], None] | None = None,
    ) -> IngestOutcome:
        if not path.exists() or not path.is_file():
            raise IngestionError(f"File does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise IngestionError(f"Unsupported file format: {path.suffix}")

        absolute_path = path.expanduser().resolve()
        file_hash = sha256_file(absolute_path)
        existing = self.db.get_book_by_path(str(absolute_path))

        if existing and not force and existing["file_hash"] == file_hash:
            return IngestOutcome(status="skipped", reason="unchanged")

        fragments, metadata = extract_book(absolute_path)
        chunks = chunk_fragments(fragments)
        if not chunks:
            raise IngestionError(f"No text content extracted from file: {absolute_path}")

        if dry_run:
            status = "would-update" if existing else "would-ingest"
            return IngestOutcome(
                status=status,
                chunks=len(chunks),
                title=metadata.get("title") if isinstance(metadata.get("title"), str) else None,
            )

        vectors = self.embeddings.embed_texts(
            [chunk.content for chunk in chunks], progress_callback=embed_progress_callback
        )

        try:
            with self.db.transaction():
                book_id = self.db.upsert_book(
                    title=str(metadata.get("title") or absolute_path.stem),
                    author=(
                        str(metadata["author"]) if metadata.get("author") is not None else None
                    ),
                    file_path=str(absolute_path),
                    file_hash=file_hash,
                    file_format=absolute_path.suffix.lower().lstrip("."),
                    file_size_bytes=absolute_path.stat().st_size,
                    total_pages=(
                        int(metadata["total_pages"])
                        if metadata.get("total_pages") is not None
                        else None
                    ),
                    existing_id=existing["id"] if existing else None,
                )
                if existing:
                    self.db.delete_chunks_for_book(book_id)
                self.db.insert_chunks(book_id, chunks, vectors)
        except Exception as exc:  # pragma: no cover - DB backend errors are integration-level
            raise IngestionError(f"Failed to ingest {absolute_path}: {exc}") from exc

        return IngestOutcome(
            status="ingested",
            chunks=len(chunks),
            title=metadata.get("title") if isinstance(metadata.get("title"), str) else None,
        )

    def scan_directory(
        self,
        root: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
        per_file_callback: Callable[[Path, IngestOutcome | None], None] | None = None,
    ) -> tuple[ScanResult, list[tuple[Path, str]]]:
        files = discover_ebook_files(root)
        ingested = 0
        skipped = 0
        failed = 0
        failures: list[tuple[Path, str]] = []

        for path in files:
            was_skipped = False
            try:
                outcome = self.ingest_file(path, dry_run=dry_run, force=force)
                if outcome.status in {"ingested", "would-ingest", "would-update"}:
                    ingested += 1
                elif outcome.status == "skipped":
                    skipped += 1
                    was_skipped = True
                if per_file_callback:
                    per_file_callback(path, outcome)
            except Exception as exc:
                failed += 1
                failures.append((path, str(exc)))
                if per_file_callback:
                    per_file_callback(path, None)
            # Throttle between books to avoid API rate limits
            if not was_skipped:
                time.sleep(2)

        return (
            ScanResult(
                scanned_files=len(files),
                ingested_files=ingested,
                skipped_files=skipped,
                failed_files=failed,
            ),
            failures,
        )
