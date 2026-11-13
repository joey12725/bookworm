# Configuration

Bookworm reads environment variables, plus a `.env` file in the working directory if one
exists. Every setting has a default, so nothing is required. `bookworm config` prints the
effective values with secrets masked. `--db PATH` on the command line overrides
`BOOKWORM_DB`.

## Library

| Variable | Default | Meaning |
|---|---|---|
| `BOOKWORM_DB` | `./bookworm.duckdb` | Database file |
| `BOOKWORM_LIBRARY` | unset | Folder `bookworm index` scans when given no paths |
| `BOOKWORM_EXCLUDE_DIRS` | unset | Comma-separated folder names to skip (hidden folders are always skipped) |

## Embeddings

| Variable | Default | Meaning |
|---|---|---|
| `BOOKWORM_EMBEDDER` | `hash` | `hash`, `fastembed`, `openai`, `openrouter`, `ollama` or `openai-compatible` |
| `BOOKWORM_EMBED_MODEL` | per provider | Model name |
| `BOOKWORM_EMBED_BASE_URL` | per provider | Base URL of an OpenAI-compatible API (`.../v1`) |
| `BOOKWORM_EMBED_API_KEY` | provider key | Falls back to `OPENAI_API_KEY` or `OPENROUTER_API_KEY` |
| `BOOKWORM_HASH_DIM` | `512` | Dimensions for the hashing embedder |

Provider defaults:

| Provider | Base URL | Embedding model | Chat model | Key variable |
|---|---|---|---|---|
| `openai` | `https://api.openai.com/v1` | `text-embedding-3-small` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `openai/text-embedding-3-small` | `openai/gpt-4o-mini` | `OPENROUTER_API_KEY` |
| `ollama` | `http://localhost:11434/v1` | `nomic-embed-text` | `llama3.2` | none |

`fastembed` defaults to `BAAI/bge-small-en-v1.5` and downloads it on first use.

Requests to API providers are batched (64 texts) and retried with exponential backoff on
rate limits, server errors and timeouts. Authentication errors fail immediately.

## Answers

| Variable | Default | Meaning |
|---|---|---|
| `BOOKWORM_CHAT_PROVIDER` | unset | `openai`, `openrouter`, `ollama` or `openai-compatible`. Unset means extractive answers |
| `BOOKWORM_CHAT_MODEL` | per provider | Model name |
| `BOOKWORM_CHAT_BASE_URL` | per provider | Base URL of an OpenAI-compatible API |
| `BOOKWORM_CHAT_API_KEY` | provider key | Falls back to the provider's key variable |

## Chunking and ranking

| Variable | Default | Meaning |
|---|---|---|
| `BOOKWORM_CHUNK_TOKENS` | `320` | Target passage size in (estimated) tokens |
| `BOOKWORM_CHUNK_OVERLAP` | `64` | Tokens repeated from the previous passage |
| `BOOKWORM_MMR_LAMBDA` | `1.0` | Below 1 turns on MMR diversity re-ranking (0.7 is a good start) |

Changing the chunk settings makes the next `bookworm index` run re-chunk every book it
scans. Vectors are reused for passages whose text did not change.

## Examples

Local semantic search, no API:

```bash
pip install 'bookworm[fastembed]'
export BOOKWORM_EMBEDDER=fastembed
bookworm reindex
```

Everything through Ollama:

```bash
ollama pull nomic-embed-text && ollama pull llama3.2
export BOOKWORM_EMBEDDER=ollama BOOKWORM_CHAT_PROVIDER=ollama
bookworm reindex
```

LM Studio or any other OpenAI-compatible server:

```bash
export BOOKWORM_EMBEDDER=openai-compatible
export BOOKWORM_EMBED_BASE_URL=http://localhost:1234/v1
export BOOKWORM_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5
```
