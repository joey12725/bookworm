#!/usr/bin/env python3
from __future__ import annotations

import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from bookworm_core.config import BookwormConfig
from bookworm_core.db import Database
from bookworm_core.embeddings import EmbeddingService
from bookworm_core.errors import BookwormError, OpenRouterError
from bookworm_core.goodreads import GoodreadsImporter
from bookworm_core.ingestion import IngestionService, discover_ebook_files
from bookworm_core.library import LibraryService, backup_database, restore_database
from bookworm_core.openrouter import OpenRouterClient
from bookworm_core.qa import QAService
from bookworm_core.search import SearchService

console = Console()


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration (e.g., '3m12s', '1h23m45s')."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m{secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins}m{secs}s"


@dataclass(slots=True)
class AppServices:
    config: BookwormConfig
    db: Database
    embeddings: EmbeddingService


@contextmanager
def build_services() -> AppServices:
    config = BookwormConfig.from_env()
    config.ensure_db_parent()
    db = Database(config.db_path)
    api_key, _ = config.require_openrouter()
    embeddings = EmbeddingService(api_key=api_key, model=config.embedding_model)
    try:
        yield AppServices(config=config, db=db, embeddings=embeddings)
    finally:
        db.close()


def _handle_error(exc: Exception) -> None:
    if isinstance(exc, click.ClickException):
        raise exc
    if isinstance(exc, (BookwormError, OpenRouterError, FileNotFoundError, ValueError, RuntimeError)):
        raise click.ClickException(str(exc)) from exc
    raise exc


def _print_search_results(title: str, results) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Book")
    table.add_column("Location")
    table.add_column("Passage", overflow="fold")

    for i, item in enumerate(results, start=1):
        location = (
            f"Chapter {item.chapter_number}" if item.chapter_number else ""
        )
        if item.chapter_title:
            location = item.chapter_title
        if item.page_number:
            location = f"{location} | p.{item.page_number}" if location else f"p.{item.page_number}"

        excerpt = textwrap.shorten(item.content, width=280, placeholder="...")
        table.add_row(str(i), f"{item.score:.4f}", item.book_title, location or "-", excerpt)

    console.print(table)


@click.group(help="Bookworm: local-first ebook ingestion, vector search, and Q&A.")
def cli() -> None:
    pass


@cli.command("scan")
@click.option("--path", "scan_path", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--dry-run", is_flag=True, help="Show what would be ingested without writing to DB.")
@click.option("--force", is_flag=True, help="Re-ingest files even if already in the database.")
@click.option("--log-file", "log_file_path", type=click.Path(path_type=Path), default=None, help="Write per-book progress logs to file.")
@click.option("--verbose", is_flag=True, help="Print per-book status lines to stderr.")
def scan_command(scan_path: Path | None, dry_run: bool, force: bool, log_file_path: Path | None, verbose: bool) -> None:
    """Recursively scan a directory and ingest supported book files."""
    try:
        with build_services() as app:
            root = scan_path or app.config.require_library_path()
            root = root.expanduser().resolve()
            if not root.exists():
                raise click.ClickException(f"Scan path does not exist: {root}")

            files = discover_ebook_files(root)
            if not files:
                console.print(f"[yellow]No PDF/EPUB files found under[/yellow] {root}")
                return

            ingestion = IngestionService(app.db, app.embeddings)

            ingested = 0
            skipped = 0
            failed = 0
            failures: list[tuple[str, str]] = []

            # Open log file if specified
            log_file = None
            if log_file_path:
                log_file = open(log_file_path, "w", buffering=1)  # Line buffered

            # Timing and ETA tracking
            import time
            start_time = time.time()
            processed_times: list[float] = []  # Times for non-skipped books only

            columns = [
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
            ]

            try:
                with Progress(*columns, console=console) as progress:
                    task = progress.add_task("Scanning books...", total=len(files))
                    for idx, file_path in enumerate(files, start=1):
                        book_start = time.time()
                        progress.update(task, description=f"Processing {file_path.name}")
                        
                        status_str = ""
                        chunks_count = 0
                        
                        try:
                            outcome = ingestion.ingest_file(
                                file_path,
                                dry_run=dry_run,
                                force=force,
                            )
                            if outcome.status == "skipped":
                                skipped += 1
                                status_str = "skipped"
                                chunks_count = 0
                            else:
                                ingested += 1
                                status_str = outcome.status
                                chunks_count = outcome.chunks
                                # Track processing time only for non-skipped books
                                book_elapsed = time.time() - book_start
                                processed_times.append(book_elapsed)
                        except Exception as exc:
                            failed += 1
                            failures.append((str(file_path), str(exc)))
                            status_str = "failed"
                            chunks_count = 0
                        finally:
                            progress.advance(task)
                        
                        # Calculate elapsed and ETA
                        total_elapsed = time.time() - start_time
                        elapsed_str = _format_duration(total_elapsed)
                        
                        # ETA calculation based on average time per non-skipped book
                        eta_str = "calculating..."
                        if processed_times:
                            avg_time_per_book = sum(processed_times) / len(processed_times)
                            books_remaining = len(files) - idx
                            # Estimate that some portion of remaining books will be skipped
                            # For simplicity, assume same skip rate as so far
                            total_processed = ingested + failed
                            if total_processed > 0:
                                skip_rate = skipped / (idx)
                                estimated_non_skipped_remaining = books_remaining * (1 - skip_rate)
                                eta_seconds = avg_time_per_book * estimated_non_skipped_remaining
                                eta_str = _format_duration(eta_seconds)
                        
                        # Format log line
                        log_line = f"[{idx}/{len(files)}] {status_str} \"{file_path.name}\" ({chunks_count} chunks) | {elapsed_str} elapsed | ETA {eta_str}\n"
                        
                        # Write to log file if specified
                        if log_file:
                            log_file.write(log_line)
                            log_file.flush()
                        
                        # Print to stderr if verbose
                        if verbose:
                            sys.stderr.write(log_line)
                            sys.stderr.flush()
            finally:
                if log_file:
                    log_file.close()

            table = Table(title="Scan Summary")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            table.add_row("Path", str(root))
            table.add_row("Files discovered", str(len(files)))
            table.add_row("Ingested", str(ingested))
            table.add_row("Skipped", str(skipped))
            table.add_row("Failed", str(failed))
            table.add_row("Mode", "DRY RUN" if dry_run else "WRITE")
            console.print(table)

            if failures:
                err_table = Table(title="Failed Files")
                err_table.add_column("File", overflow="fold")
                err_table.add_column("Error", overflow="fold")
                for file_name, err in failures:
                    err_table.add_row(file_name, textwrap.shorten(err, width=240, placeholder="..."))
                console.print(err_table)
                raise click.ClickException(f"{failed} files failed during scan.")

    except Exception as exc:
        _handle_error(exc)


@cli.command("add")
@click.argument("book_path", type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.option("--force", is_flag=True, help="Re-ingest even if file hash matches existing record.")
def add_command(book_path: Path, force: bool) -> None:
    """Add a single PDF or EPUB file."""
    try:
        with build_services() as app:
            ingestion = IngestionService(app.db, app.embeddings)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Ingesting book...", total=None)
                outcome = ingestion.ingest_file(book_path, force=force)
                progress.update(task, description="Done")

            if outcome.status == "skipped":
                console.print(f"[yellow]Skipped[/yellow] {book_path} (unchanged)")
                return

            table = Table(title="Book Added")
            table.add_column("Field")
            table.add_column("Value")
            table.add_row("File", str(book_path.expanduser().resolve()))
            table.add_row("Title", outcome.title or book_path.stem)
            table.add_row("Chunks", str(outcome.chunks))
            table.add_row("Status", outcome.status)
            console.print(table)

    except Exception as exc:
        _handle_error(exc)


@cli.command("search-vector")
@click.argument("query")
@click.option("--limit", type=click.IntRange(1, 100), default=5, show_default=True)
@click.option("--book", "book_filter", default=None, help="Filter by book title (substring).")
def search_vector_command(query: str, limit: int, book_filter: str | None) -> None:
    """Semantic vector search across chunk embeddings."""
    try:
        with build_services() as app:
            service = SearchService(app.db, app.embeddings)
            results = service.search_vector(query=query, limit=limit, book_filter=book_filter)
            if not results:
                console.print("[yellow]No matches found.[/yellow]")
                return
            _print_search_results("Vector Search Results", results)
    except Exception as exc:
        _handle_error(exc)


@cli.command("search-keyword")
@click.argument("query")
@click.option("--limit", type=click.IntRange(1, 100), default=5, show_default=True)
@click.option("--book", "book_filter", default=None, help="Filter by book title (substring).")
def search_keyword_command(query: str, limit: int, book_filter: str | None) -> None:
    """Keyword search across chunk text."""
    try:
        with build_services() as app:
            service = SearchService(app.db, app.embeddings)
            results = service.search_keyword(query=query, limit=limit, book_filter=book_filter)
            if not results:
                console.print("[yellow]No matches found.[/yellow]")
                return
            _print_search_results("Keyword Search Results", results)
    except Exception as exc:
        _handle_error(exc)


@cli.command("import-goodreads")
@click.argument("csv_path", type=click.Path(exists=True, path_type=Path, dir_okay=False))
def import_goodreads_command(csv_path: Path) -> None:
    """Import Goodreads CSV data and enrich matching books."""
    try:
        with build_services() as app:
            importer = GoodreadsImporter(app.db)
            result, unmatched = importer.import_csv(csv_path)

            table = Table(title="Goodreads Import")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            table.add_row("CSV path", str(csv_path.expanduser().resolve()))
            table.add_row("Rows processed", str(result.total_rows))
            table.add_row("Matched", str(result.matched_rows))
            table.add_row("Unmatched", str(result.unmatched_rows))
            console.print(table)

            if unmatched:
                preview = unmatched[:10]
                unmatched_table = Table(title="Unmatched Preview (first 10)")
                unmatched_table.add_column("Title")
                unmatched_table.add_column("Author")
                unmatched_table.add_column("Best score", justify="right")
                for row in preview:
                    score = f"{row['score']:.2f}" if row["score"] is not None else "-"
                    unmatched_table.add_row(row["title"], row["author"], score)
                console.print(unmatched_table)

    except Exception as exc:
        _handle_error(exc)


@cli.command("get-book")
@click.argument("title")
def get_book_command(title: str) -> None:
    """Show metadata and ingestion stats for a specific book."""
    try:
        with build_services() as app:
            library = LibraryService(app.db)
            summary = library.get_book_summary(title)
            if not summary:
                raise click.ClickException(f"No book found matching: {title}")

            table = Table(title=f"Book: {summary['title']}")
            table.add_column("Field")
            table.add_column("Value")
            table.add_row("Author", summary.get("author") or "Unknown")
            table.add_row("Format", summary.get("format") or "-")
            table.add_row("File path", summary.get("file_path") or "-")
            table.add_row("File size", str(summary.get("file_size_bytes") or 0))
            table.add_row("Pages/sections", str(summary.get("total_pages") or "-"))
            table.add_row("Chapter count", str(summary.get("chapter_count") or 0))
            table.add_row("Chunk count", str(summary.get("chunk_count") or 0))
            table.add_row("Goodreads rating", str(summary.get("goodreads_rating") or "-"))
            table.add_row("Goodreads status", summary.get("goodreads_read_status") or "-")
            table.add_row("Goodreads shelves", summary.get("goodreads_shelves") or "-")
            table.add_row("Added at", str(summary.get("added_at")))
            table.add_row("Updated at", str(summary.get("updated_at")))
            console.print(table)

    except Exception as exc:
        _handle_error(exc)


@cli.command("get-chapter")
@click.argument("title")
@click.argument("chapter_number", type=int)
def get_chapter_command(title: str, chapter_number: int) -> None:
    """Get full text for a chapter from an EPUB-backed book."""
    try:
        with build_services() as app:
            library = LibraryService(app.db)
            found = library.get_chapter_text(title, chapter_number)
            if not found:
                raise click.ClickException(
                    f"No chapter {chapter_number} found for book matching: {title}"
                )

            book, chapter_text = found
            console.print(
                Panel(
                    chapter_text,
                    title=f"{book['title']} - Chapter {chapter_number}",
                    expand=True,
                )
            )
    except Exception as exc:
        _handle_error(exc)


@cli.command("stats")
def stats_command() -> None:
    """Show library-level ingestion and metadata statistics."""
    try:
        with build_services() as app:
            stats = app.db.get_stats()

            totals = Table(title="Library Totals")
            totals.add_column("Metric")
            totals.add_column("Value", justify="right")
            totals.add_row("Books", str(stats["total_books"]))
            totals.add_row("Chunks", str(stats["total_chunks"]))
            totals.add_row("Embeddings", str(stats["total_embeddings"]))
            console.print(totals)

            by_format = Table(title="Books by Format")
            by_format.add_column("Format")
            by_format.add_column("Count", justify="right")
            if stats["by_format"]:
                for fmt, count in stats["by_format"]:
                    by_format.add_row(fmt or "unknown", str(count))
            else:
                by_format.add_row("-", "0")
            console.print(by_format)

            authors = Table(title="Top Authors")
            authors.add_column("Author")
            authors.add_column("Books", justify="right")
            if stats["top_authors"]:
                for author, count in stats["top_authors"]:
                    authors.add_row(author, str(count))
            else:
                authors.add_row("-", "0")
            console.print(authors)

            recent = Table(title="Recently Added")
            recent.add_column("Title")
            recent.add_column("Author")
            recent.add_column("Added At")
            if stats["recent"]:
                for title, author, added_at in stats["recent"]:
                    recent.add_row(title, author or "Unknown", str(added_at))
            else:
                recent.add_row("-", "-", "-")
            console.print(recent)

    except Exception as exc:
        _handle_error(exc)


@cli.command("ask")
@click.argument("question")
@click.option("--limit", type=click.IntRange(1, 20), default=8, show_default=True)
@click.option("--book", "book_filter", default=None, help="Filter retrieval to a specific book.")
def ask_command(question: str, limit: int, book_filter: str | None) -> None:
    """Answer a question using retrieved passages and OpenRouter synthesis."""
    try:
        with build_services() as app:
            api_key, model = app.config.require_openrouter()
            search = SearchService(app.db, app.embeddings)
            qa = QAService(search, OpenRouterClient(api_key=api_key, model=model))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Retrieving relevant passages...", total=None)
                result = qa.ask(question, limit=limit, book_filter=book_filter)
                progress.update(task, description="Generating grounded answer...")

            console.print(Panel(result.answer, title="Answer", expand=True))
            _print_search_results("Sources", result.sources)

    except Exception as exc:
        _handle_error(exc)


@cli.command("backup")
@click.argument("destination", type=click.Path(path_type=Path))
def backup_command(destination: Path) -> None:
    """Create a file backup of the DuckDB database."""
    try:
        config = BookwormConfig.from_env()
        if not config.db_path.exists():
            raise click.ClickException(f"Database file not found: {config.db_path}")
        output = backup_database(config.db_path, destination)
        console.print(f"[green]Backup created:[/green] {output}")
    except Exception as exc:
        _handle_error(exc)


@cli.command("restore")
@click.argument("source", type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.option("--yes", is_flag=True, help="Skip overwrite confirmation prompt.")
def restore_command(source: Path, yes: bool) -> None:
    """Restore the DuckDB database from a backup file."""
    try:
        config = BookwormConfig.from_env()
        config.ensure_db_parent()

        if config.db_path.exists() and not yes:
            proceed = click.confirm(
                f"Restore will overwrite {config.db_path}. Continue?",
                default=False,
            )
            if not proceed:
                console.print("Restore cancelled.")
                return

        output = restore_database(source, config.db_path)
        console.print(f"[green]Database restored:[/green] {output}")
    except Exception as exc:
        _handle_error(exc)


if __name__ == "__main__":
    try:
        cli()
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        sys.exit(exc.exit_code)
