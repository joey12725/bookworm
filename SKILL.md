# Bookworm Skill

Use this skill to ingest local ebooks into DuckDB, run semantic/keyword search, and answer grounded questions from the library.

## What This Skill Provides

- Local ingestion of PDF/EPUB books
- Embedding generation through OpenRouter `openai/text-embedding-3-small`
- Vector + keyword retrieval over chunks in DuckDB
- Goodreads metadata import and library stats
- OpenRouter-grounded Q&A over retrieved passages

## Setup

1. Create and activate a virtual environment:
   - `python -m venv venv`
   - `source venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure `.env` from `.env.example`:
   - `BOOKWORM_LIBRARY_PATH`
   - `BOOKWORM_DB_PATH`
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_MODEL`

## Commands

- `python bookworm.py scan --path /path/to/books`
- `python bookworm.py add /path/to/book.pdf`
- `python bookworm.py search-vector "your semantic query" --limit 5`
- `python bookworm.py search-keyword "exact phrase" --limit 5`
- `python bookworm.py import-goodreads /path/to/goodreads_export.csv`
- `python bookworm.py get-book "Book Title"`
- `python bookworm.py get-chapter "Book Title" 5`
- `python bookworm.py stats`
- `python bookworm.py ask "Question grounded in my books"`
- `python bookworm.py backup ./backups/bookworm.duckdb`
- `python bookworm.py restore ./backups/bookworm.duckdb --yes`

## Notes for Agents

- Prefer `scan --dry-run` before large ingestions.
- Use `--force` when a file changed but retained the same path.
- `search-vector` and `ask` require a working `OPENROUTER_API_KEY` because query embeddings are generated through OpenRouter.
- `search-keyword`, `get-book`, `get-chapter`, and `stats` work offline once the DuckDB index exists.
