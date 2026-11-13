from __future__ import annotations

import os
import shutil
from pathlib import Path

from bookworm import Bookworm, Settings
from bookworm.embeddings import HashingEmbedder

from .conftest import CountingEmbedder


def statuses(report) -> dict[str, str]:
    return {Path(f.path).name: f.status for f in report.files}


def test_first_run_adds_everything(bw: Bookworm, embedder: CountingEmbedder) -> None:
    books = bw.books()
    assert [b.title for b in books] == [
        "Alice's Adventures in Wonderland",
        "Pride and Prejudice",
        "The Art of War",
    ]
    total = sum(b.chunks for b in books)
    assert embedder.embedded == total
    assert bw.stats()["chunks"] == total


def test_second_run_skips_unchanged_files(
    bw: Bookworm, embedder: CountingEmbedder, library: Path
) -> None:
    before = embedder.embedded
    report = bw.index([library])
    assert set(statuses(report).values()) == {"unchanged"}
    assert embedder.embedded == before


def test_touched_file_is_rehashed_not_reembedded(
    bw: Bookworm, embedder: CountingEmbedder, library: Path
) -> None:
    path = library / "pride-and-prejudice.txt"
    os.utime(path, (1_000_000, 1_000_000))
    before = embedder.embedded
    assert statuses(bw.index([library]))[path.name] == "unchanged"
    assert embedder.embedded == before


def test_edit_reuses_vectors_of_unchanged_passages(
    bw: Bookworm, embedder: CountingEmbedder, library: Path
) -> None:
    path = library / "pride-and-prejudice.txt"
    path.write_text(path.read_text() + "\n\nA brand new closing paragraph about Netherfield.\n")
    before = embedder.embedded
    report = bw.index([library])
    result = next(f for f in report.files if f.path.endswith(path.name))
    assert result.status == "updated"
    assert result.reused_vectors >= result.chunks - 2
    assert embedder.embedded - before <= 2
    assert bw.search('"brand new closing paragraph"', mode="keyword")


def test_moved_file_is_relinked(bw: Bookworm, embedder: CountingEmbedder, library: Path) -> None:
    (library / "classics").mkdir()
    new = library / "classics" / "alice.md"
    shutil.move(library / "alice-in-wonderland.md", new)
    before = embedder.embedded
    report = bw.index([library])
    assert statuses(report)["alice.md"] == "moved"
    assert embedder.embedded == before
    assert bw.find_book("alice").path == str(new.resolve())


def test_prune_removes_deleted_books(bw: Bookworm, library: Path) -> None:
    (library / "the-art-of-war.epub").unlink()
    report = bw.index([library], prune=True)
    assert report.removed and report.removed[0].endswith("the-art-of-war.epub")
    assert len(bw.books()) == 2
    assert not bw.search("Sun Tzu", mode="keyword")


def test_dry_run_writes_nothing(settings: Settings, library: Path) -> None:
    with Bookworm(settings, embedder=HashingEmbedder(64)) as bw:
        report = bw.index([library], dry_run=True)
        assert set(statuses(report).values()) == {"would-add"}
        assert bw.books() == []


def test_bad_file_is_reported_and_others_continue(bw: Bookworm, library: Path) -> None:
    (library / "broken.epub").write_bytes(b"garbage")
    report = bw.index([library])
    assert statuses(report)["broken.epub"] == "failed"
    assert report.count("failed") == 1
    assert report.count("unchanged") == 3


def test_excluded_and_hidden_dirs_are_skipped(settings: Settings, library: Path) -> None:
    for name in ("comics", ".trash"):
        (library / name).mkdir()
        (library / name / "skip.txt").write_text("Skip me please.")
    with Bookworm(
        settings.with_overrides(excluded_dirs=("comics",)), embedder=HashingEmbedder(64)
    ) as bw:
        report = bw.index([library])
    assert "skip.txt" not in statuses(report)


def test_changed_chunking_rechunks(bw: Bookworm, library: Path) -> None:
    before = sum(b.chunks for b in bw.books())
    bw.settings = bw.settings.with_overrides(chunk_tokens=120, chunk_overlap=20)
    report = bw.index([library])
    assert set(statuses(report).values()) == {"updated"}
    assert sum(b.chunks for b in bw.books()) > before


def test_reindex_switches_embedder(bw: Bookworm) -> None:
    bw._embedder = HashingEmbedder(dim=64)
    n = bw.reindex()
    assert n == bw.stats()["chunks"]
    assert bw.store.get_meta("embedder") == "hash:v1:64"
    assert bw.search("rabbit", mode="vector")


def test_read_chapter_reassembles_text_without_overlap(bw: Bookworm, library: Path) -> None:
    _book, title, text = bw.read("pride", 1)
    assert title == "Chapter I"
    assert text.startswith("It is a truth universally acknowledged")
    assert text.count("It is a truth universally acknowledged") == 1
    source = (library / "pride-and-prejudice.txt").read_text()
    assert "its solace was visiting and news." in source
    assert text.rstrip().endswith("its solace was visiting and news.")


def test_chapters_listing(bw: Bookworm) -> None:
    _book, chapters = bw.chapters("The Art of War")
    assert [title for _, title, _ in chapters] == [
        "Chapter I. Laying Plans",
        "Chapter II. Waging War",
        "Chapter III. Attack By Stratagem",
    ]


def test_remove_book(bw: Bookworm) -> None:
    removed = bw.remove("Art of War")
    assert removed.title == "The Art of War"
    assert len(bw.books()) == 2


def test_backup_and_restore(bw: Bookworm, tmp_path: Path) -> None:
    from bookworm.app import restore

    backup = bw.backup(tmp_path / "backups")
    assert backup.exists() and backup.stat().st_size > 0
    copy = tmp_path / "restored.duckdb"
    restore(backup, copy)
    with Bookworm(Settings(db_path=copy), embedder=HashingEmbedder(256)) as other:
        assert len(other.books()) == 3
