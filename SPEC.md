# Bookworm - Local Books Vector Library

A local-first book ingestion and semantic search system. Python CLI backed by DuckDB with vector similarity search.

## Overview

Bookworm scans a local ebook library, extracts text, chunks it, generates vector embeddings, and stores everything in DuckDB. Users can then semantically search across their entire library, find passages by meaning, and get AI-powered answers grounded in their books.

Current implementation note: query and ingestion embeddings are generated through OpenRouter's embeddings API using `openai/text-embedding-3-small`. The core library and DuckDB index stay local, but vector search depends on a working `OPENROUTER_API_KEY`.

## Tech Stack

- **Language:** Python 3.11+
- **Database:** DuckDB with VSS (Vector Similarity Search) extension
- **Embeddings:** OpenRouter embeddings API, default `openai/text-embedding-3-small` (1536 dimensions)
- **Text extraction:** PyMuPDF (fitz) for PDF, ebooklib for EPUB
- **AI:** OpenRouter API for embeddings and Q&A synthesis
- **CLI framework:** argparse or click

## Commands

### `scan`
Recursively scan a directory for ebook files (PDF, EPUB). Extract text, chunk, embed, and store.
- `--path` - Directory to scan (default: configured library path)
- `--dry-run` - Show what would be ingested without doing it
- `--force` - Re-ingest even if file already exists in DB
- Tracks file hash to avoid re-processing unchanged files

### `add`
Add a single book file.
- `bookworm add /path/to/book.pdf`

### `search-vector`
Semantic search across all books using vector similarity.
- `bookworm search-vector "first impressions and pride"`
- Returns top N passages with book title, chapter, page number, relevance score
- `--limit N` - Number of results (default: 5)
- `--book` - Filter to specific book

### `search-keyword`
Traditional keyword/full-text search.
- `bookworm search-keyword "waitlist"`
- `--limit N`
- `--book` - Filter to specific book

### `import-goodreads`
Import a Goodreads CSV export to enrich book metadata (read status, ratings, shelves).
- `bookworm import-goodreads goodreads_export.csv`
- Matches by title/author fuzzy matching
- Updates existing records, doesn't duplicate

### `get-book`
Show metadata and stats for a specific book.
- `bookworm get-book "Pride and Prejudice"`
- Fuzzy title matching

### `get-chapter`
Get full text of a specific chapter.
- `bookworm get-chapter "Pride and Prejudice" 5`

### `stats`
Show library statistics.
- Total books, total chunks, total embeddings
- Books by format, top authors, recently added

### `ask`
Ask a question and get an AI-generated answer grounded in your library.
- `bookworm ask "What are the author's main arguments in chapter 3?"`
- Uses vector search to find relevant passages, then sends to OpenRouter for synthesis
- Shows source citations

### `backup` / `restore`
Export/import the DuckDB database.

## Database Schema

### `books` table
- id (UUID, primary key)
- title (TEXT)
- author (TEXT)
- file_path (TEXT)
- file_hash (TEXT, for dedup)
- format (TEXT: pdf, epub)
- file_size_bytes (INTEGER)
- total_pages (INTEGER)
- goodreads_rating (FLOAT, nullable)
- goodreads_read_status (TEXT, nullable)
- goodreads_shelves (TEXT, nullable)
- added_at (TIMESTAMP)
- updated_at (TIMESTAMP)

### `chunks` table
- id (UUID, primary key)
- book_id (UUID, FK to books)
- chapter_title (TEXT, nullable)
- chapter_number (INTEGER, nullable)
- page_number (INTEGER, nullable)
- chunk_index (INTEGER) - position within the book
- content (TEXT) - the actual text
- token_count (INTEGER)
- embedding (FLOAT[1536]) - vector embedding

### Chunking Strategy
- Split by paragraphs, merge small paragraphs until ~500 tokens
- Overlap: 50 tokens between chunks for context continuity
- Preserve chapter/page boundaries where possible

## Configuration

Environment variables (`.env` file):
```
BOOKWORM_LIBRARY_PATH=/path/to/ebooks
BOOKWORM_DB_PATH=./bookworm.duckdb
OPENROUTER_API_KEY=<key>
OPENROUTER_MODEL=openai/gpt-4o-mini
```

## Installation

```bash
cd bookworm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Copy .env.example to .env and configure
```

## Requirements

```
duckdb>=1.0.0
PyMuPDF>=1.24.0
ebooklib>=0.18
python-dotenv>=1.0.0
httpx>=0.27.0
click>=8.0.0
rich>=13.0.0
```

## Key Design Decisions

- **Local-first storage:** Book text, chunks, and embeddings live in local DuckDB; embedding generation currently uses OpenRouter
- **DuckDB:** Single-file database, no server needed, excellent for vector search via VSS extension
- **OpenRouter embeddings:** The current code uses `openai/text-embedding-3-small`; stale or missing OpenRouter auth blocks new query embeddings and therefore vector search
- **Grounded Q&A:** The `ask` command uses vector search for retrieval, then OpenRouter for answer synthesis
- **Rich output:** Use rich library for pretty terminal output (tables, progress bars, colored text)
