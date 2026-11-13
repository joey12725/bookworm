"""The ``bookworm`` command line."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from bookworm import __version__
from bookworm.app import Bookworm, restore
from bookworm.config import Settings
from bookworm.errors import BookwormError
from bookworm.models import FileResult, Hit, SearchMode
from bookworm.text import snippet

console = Console(highlight=False)

STATUS_STYLE = {
    "added": "green",
    "updated": "cyan",
    "moved": "blue",
    "unchanged": "dim",
    "would-add": "green",
    "would-update": "cyan",
    "failed": "red",
}


def _settings(ctx: click.Context) -> Settings:
    return cast(Settings, ctx.obj)


@contextmanager
def _open(ctx: click.Context) -> Iterator[Bookworm]:
    try:
        with Bookworm.open(_settings(ctx)) as bw:
            yield bw
    except BookwormError as exc:
        raise click.ClickException(str(exc)) from exc


def _as_dict(obj: Any) -> dict[str, Any]:
    return {k: getattr(obj, k) for k in obj.__slots__}


def _print_json(data: Any) -> None:
    click.echo(json.dumps(data, indent=2, default=str, ensure_ascii=False))


def _hits_table(hits: list[Hit], title: str, query: str) -> Table:
    table = Table(title=title, title_justify="left", show_lines=True, expand=True)
    table.add_column("#", style="bold cyan", justify="right", width=2)
    table.add_column("Source", style="magenta", ratio=1)
    table.add_column("Passage", ratio=3)
    for n, hit in enumerate(hits, start=1):
        table.add_row(str(n), hit.citation.format(), snippet(hit.text, query))
    return table


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--db",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Library database file. [env: BOOKWORM_DB, default: ./bookworm.duckdb]",
)
@click.version_option(__version__, prog_name="bookworm")
@click.pass_context
def cli(ctx: click.Context, db: Path | None) -> None:
    """Bookworm: a searchable, citable knowledge base for your ebook library."""
    settings = Settings.from_env()
    ctx.obj = settings.with_overrides(db_path=db)


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Show what would change without writing.")
@click.option("--force", is_flag=True, help="Re-extract files even if they look unchanged.")
@click.option("--prune", is_flag=True, help="Remove books whose files were deleted.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.pass_context
def index(
    ctx: click.Context,
    paths: tuple[Path, ...],
    dry_run: bool,
    force: bool,
    prune: bool,
    as_json: bool,
) -> None:
    """Add or update books from files and folders (EPUB, PDF, Markdown, TXT, HTML, MOBI).

    Re-running is cheap: unchanged files are skipped, moved files are re-linked, and
    edited books reuse the embeddings of every unchanged passage.
    """
    with _open(ctx) as bw:
        if as_json:
            report = bw.index(paths, dry_run=dry_run, force=force, prune=prune)
        else:
            with Progress(
                TextColumn("[bold]Indexing"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TextColumn("{task.description}", style="dim"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("", total=None)

                def on_file(result: FileResult, n: int, total: int) -> None:
                    progress.update(
                        task, completed=n, total=total, description=Path(result.path).name
                    )
                    style = STATUS_STYLE.get(result.status, "")
                    detail = f"{result.chunks} chunks" if result.chunks else ""
                    if result.reused_vectors:
                        detail += f", {result.reused_vectors} reused"
                    if result.error:
                        detail = result.error
                    progress.console.print(
                        Text.assemble(
                            (f"{result.status:>12} ", style),
                            (result.title or Path(result.path).name, "bold"),
                            (f"  {detail}", "dim"),
                        )
                    )

                report = bw.index(
                    paths, dry_run=dry_run, force=force, prune=prune, progress=on_file
                )

    if as_json:
        _print_json({"files": [_as_dict(f) for f in report.files], "removed": report.removed})
    else:
        for path in report.removed:
            console.print(Text.assemble((f"{'removed':>12} ", "yellow"), path))
        summary = ", ".join(
            f"{report.count(s)} {s}"
            for s in (
                "added",
                "updated",
                "moved",
                "unchanged",
                "would-add",
                "would-update",
                "failed",
            )
            if report.count(s)
        )
        console.print(f"[bold]Done.[/bold] {summary or 'No supported files found.'}")
    if report.count("failed"):
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("-k", "--limit", type=click.IntRange(1, 100), default=5, show_default=True)
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "keyword", "vector"]),
    default="hybrid",
    show_default=True,
    help="hybrid fuses BM25 and vector rankings; keyword is BM25 only.",
)
@click.option("--book", default=None, help="Only search books whose title matches.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.pass_context
def search(
    ctx: click.Context, query: str, limit: int, mode: str, book: str | None, as_json: bool
) -> None:
    """Search passages. Put "exact phrases" in double quotes."""
    with _open(ctx) as bw:
        hits = bw.search(query, limit=limit, mode=cast(SearchMode, mode), book=book)
    if as_json:
        _print_json([h.to_dict() for h in hits])
    elif not hits:
        console.print("[yellow]No matches.[/yellow]")
    else:
        console.print(_hits_table(hits, f'Results for "{query}"', query))


@cli.command()
@click.argument("question")
@click.option("-k", "--limit", type=click.IntRange(1, 30), default=6, show_default=True)
@click.option("--book", default=None, help="Only use books whose title matches.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.pass_context
def ask(ctx: click.Context, question: str, limit: int, book: str | None, as_json: bool) -> None:
    """Answer a question from your books, with numbered citations.

    Uses the configured chat model (BOOKWORM_CHAT_PROVIDER), or an extractive answer
    built from the best-matching sentences when none is set.
    """
    with _open(ctx) as bw:
        result = bw.ask(question, limit=limit, book=book)
    if as_json:
        _print_json(result.to_dict())
        return
    subtitle = "extractive" if result.mode == "extractive" else "grounded in the sources below"
    console.print(
        Panel(Markdown(result.text), title="Answer", subtitle=subtitle, title_align="left")
    )
    cited = [result.sources[n - 1] for n in result.cited] if result.cited else result.sources
    numbers = result.cited or list(range(1, len(result.sources) + 1))
    table = Table(
        title="Sources", title_justify="left", show_header=False, box=None, padding=(0, 1)
    )
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="magenta")
    for n, hit in zip(numbers, cited, strict=True):
        table.add_row(f"[{n}]", hit.citation.format())
    console.print(table)


@cli.command("books")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.pass_context
def books_command(ctx: click.Context, as_json: bool) -> None:
    """List the books in the library."""
    with _open(ctx) as bw:
        books = bw.books()
    if as_json:
        _print_json([_as_dict(b) for b in books])
        return
    table = Table(title=f"{len(books)} books", title_justify="left")
    for col, kw in [
        ("Title", {"style": "bold"}), ("Author", {}), ("Format", {}),
        ("Chapters", {"justify": "right"}), ("Chunks", {"justify": "right"}),
    ]:  # fmt: skip
        table.add_column(col, **kw)  # type: ignore[arg-type]
    for b in books:
        table.add_row(b.title, b.author or "-", b.format, str(b.chapters), str(b.chunks))
    console.print(table)


@cli.command()
@click.argument("title")
@click.pass_context
def show(ctx: click.Context, title: str) -> None:
    """Show a book's details and table of contents."""
    with _open(ctx) as bw:
        book, chapters = bw.chapters(title)
    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column()
    info.add_row("Author", book.author or "-")
    info.add_row("File", book.path)
    info.add_row("Format", book.format + (f", {book.pages} pages" if book.pages else ""))
    info.add_row("Indexed", f"{book.chunks} chunks, updated {book.updated_at:%Y-%m-%d %H:%M}")
    if book.rating or book.shelf:
        info.add_row("Goodreads", f"{book.rating or '-'} stars, {book.shelf or '-'}")
    console.print(Panel(info, title=f"[bold]{book.title}[/bold]", title_align="left"))
    if chapters:
        toc = Table(title="Contents", title_justify="left", box=None)
        toc.add_column("#", justify="right", style="cyan")
        toc.add_column("Chapter")
        toc.add_column("Chunks", justify="right", style="dim")
        for number, chapter_title, n in chapters:
            toc.add_row(str(number), chapter_title or "-", str(n))
        console.print(toc)


@cli.command()
@click.argument("title")
@click.argument("chapter", type=int)
@click.pass_context
def read(ctx: click.Context, title: str, chapter: int) -> None:
    """Print the full text of one chapter."""
    with _open(ctx) as bw:
        book, chapter_title, text = bw.read(title, chapter)
    heading = f"{book.title} · {chapter_title or f'Chapter {chapter}'}"
    console.print(Panel(text, title=heading, title_align="left"))


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.pass_context
def stats(ctx: click.Context, as_json: bool) -> None:
    """Library statistics."""
    with _open(ctx) as bw:
        data = bw.stats()
    if as_json:
        _print_json(data)
        return
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column(justify="right", style="bold")
    grid.add_row("Books", f"{data['books']:,}")
    grid.add_row("Passages", f"{data['chunks']:,}")
    grid.add_row("Tokens (approx.)", f"{data['tokens']:,}")
    grid.add_row("Vocabulary", f"{data['vocabulary']:,} terms")
    grid.add_row("Embedder", str(data["embedder"]))
    if "db_bytes" in data:
        grid.add_row("Database", f"{data['db_bytes'] / 1_048_576:.1f} MB")
    formats = ", ".join(f"{fmt} {n}" for fmt, n in data["by_format"]) or "-"
    grid.add_row("Formats", formats)
    console.print(Panel(grid, title="Library", title_align="left", expand=False))


@cli.command()
@click.argument("title")
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
@click.pass_context
def remove(ctx: click.Context, title: str, yes: bool) -> None:
    """Remove a book from the index (the file is not touched)."""
    with _open(ctx) as bw:
        book = bw.find_book(title)
        if not yes:
            click.confirm(f"Remove {book.title!r} from the library?", abort=True)
        bw.store.delete_book(book.id)
    console.print(f"Removed [bold]{book.title}[/bold].")


@cli.command()
@click.pass_context
def reindex(ctx: click.Context) -> None:
    """Re-embed every passage with the configured embedder (after switching models)."""
    with _open(ctx) as bw, Progress(console=console, transient=True) as progress:
        task = progress.add_task("Embedding", total=None)
        n = bw.reindex(lambda done, total: progress.update(task, completed=done, total=total))
    console.print(f"Re-embedded {n:,} passages with [bold]{bw.embedder.name}[/bold].")


@cli.command("import-goodreads")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def import_goodreads_command(ctx: click.Context, csv_path: Path) -> None:
    """Attach ratings and shelves from a Goodreads CSV export."""
    with _open(ctx) as bw:
        report = bw.import_goodreads(csv_path)
    console.print(
        f"Matched [bold]{len(report.matched)}[/bold] of {report.rows} rows to books in the library."
    )
    for title in report.unmatched[:10]:
        console.print(f"  [dim]not in library:[/dim] {title}")


@cli.command()
@click.argument("destination", type=click.Path(path_type=Path))
@click.pass_context
def backup(ctx: click.Context, destination: Path) -> None:
    """Copy the library database to DESTINATION."""
    with _open(ctx) as bw:
        out = bw.backup(destination)
    console.print(f"Backup written to [bold]{out}[/bold]")


@cli.command("restore")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--yes", is_flag=True, help="Overwrite without asking.")
@click.pass_context
def restore_command(ctx: click.Context, source: Path, yes: bool) -> None:
    """Replace the library database with a backup."""
    db_path = _settings(ctx).db_path
    if db_path.exists() and not yes:
        click.confirm(f"Overwrite {db_path}?", abort=True)
    restore(source, db_path)
    console.print(f"Restored [bold]{db_path}[/bold] from {source}")


@cli.command("config")
@click.pass_context
def config_command(ctx: click.Context) -> None:
    """Show the effective configuration (secrets masked)."""
    table = Table(box=None, show_header=False)
    table.add_column(style="dim")
    table.add_column()
    for key, value in _settings(ctx).redacted().items():
        table.add_row(key, value)
    console.print(table)


@cli.command()
@click.pass_context
def mcp(ctx: click.Context) -> None:
    """Serve the library to AI agents over the Model Context Protocol (stdio)."""
    from bookworm.mcp import serve
    from bookworm.store import Store

    settings = _settings(ctx)
    if not settings.db_path.exists():
        raise click.ClickException(
            f"No library at {settings.db_path}. Run `bookworm index` first or set BOOKWORM_DB."
        )
    try:
        # Read-only: agents can search and read, never modify, and several can share a library.
        store = Store(settings.db_path, read_only=True)
        with Bookworm(settings, store=store) as bw:
            serve(bw)
    except BookwormError as exc:
        raise click.ClickException(str(exc)) from exc


def main() -> None:
    cli(prog_name="bookworm")


if __name__ == "__main__":  # pragma: no cover
    main()
