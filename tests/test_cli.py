from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bookworm.cli import cli


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)  # no stray .env from the developer's machine
    for var in ("BOOKWORM_EMBEDDER", "BOOKWORM_CHAT_PROVIDER", "BOOKWORM_DB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BOOKWORM_HASH_DIM", "128")
    db = tmp_path / "cli.duckdb"
    runner = CliRunner()

    def invoke(*args: str, ok: bool = True, input: str | None = None):
        result = runner.invoke(cli, ["--db", str(db), *args], input=input, catch_exceptions=False)
        if ok:
            assert result.exit_code == 0, result.output
        return result

    return invoke


def test_full_cli_flow(run, library: Path, tmp_path: Path) -> None:
    out = run("index", str(library)).output
    assert "3 added" in out

    data = json.loads(run("index", str(library), "--json").output)
    assert {f["status"] for f in data["files"]} == {"unchanged"}

    hits = json.loads(run("search", "truth universally acknowledged", "--json", "-k", "2").output)
    assert hits[0]["citation"]["book"] == "Pride and Prejudice"

    assert "Prejudice" in run("search", "Netherfield", "--mode", "keyword").output

    answer = json.loads(run("ask", "What did the bottle say?", "--json").output)
    assert answer["mode"] == "extractive" and answer["sources"]
    assert "Answer" in run("ask", "Who is Bingley?").output

    assert "3 books" in run("books").output
    assert "Laying Plans" in run("show", "art of war").output
    assert "Sun Tzŭ said" in run("read", "art of war", "1").output
    stats = json.loads(run("stats", "--json").output)
    assert stats["books"] == 3 and stats["embedder"] == "hash:v1:128"
    assert "Library" in run("stats").output

    csv = tmp_path / "goodreads.csv"
    csv.write_text(
        "Title,Author,My Rating,Exclusive Shelf,Bookshelves\n"
        "Pride and Prejudice,Jane Austen,5,read,classics\n"
        "A Book Not In The Library,Nobody,3,to-read,\n"
    )
    assert "Matched 1 of 2" in run("import-goodreads", str(csv)).output
    assert "5.0 stars, read" in run("show", "pride").output

    backup = tmp_path / "backup.duckdb"
    run("backup", str(backup))
    run("remove", "alice", "--yes")
    assert "2 books" in run("books").output
    run("restore", str(backup), "--yes")
    assert "3 books" in run("books").output


def test_cli_errors(run) -> None:
    result = run("show", "missing", ok=False)
    assert result.exit_code == 1
    assert "No book matching" in result.output
    result = run("index", ok=False)
    assert "BOOKWORM_LIBRARY" in result.output


def test_config_masks_secrets(run, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKWORM_EMBED_API_KEY", "super-secret")
    out = run("config").output
    assert "super-secret" not in out
    assert "embed_api_key" in out


def test_failed_file_sets_exit_code(run, library: Path) -> None:
    (library / "broken.epub").write_bytes(b"nope")
    result = run("index", str(library), ok=False)
    assert result.exit_code == 1
    assert "failed" in result.output
