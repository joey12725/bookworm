# Architecture

Bookworm is a small pipeline with one storage layer and three front ends. Everything below
`bookworm.app.Bookworm` is plain, typed Python with no global state, so each stage can be
tested and swapped independently.

```
src/bookworm/
├── extractors/      file -> Document (sections with chapter, heading path, page)
│   ├── plaintext.py   TXT, Project Gutenberg aware
│   ├── markdown.py    front matter + ATX headings
│   ├── html.py        stdlib HTMLParser; shared by EPUB and MOBI
│   ├── epub.py        zipfile + ElementTree; spine order, nav/NCX titles
│   ├── pdf.py         pypdf; page numbers + outline chapters
│   └── mobi.py        unpack to HTML via the mobi package
├── chunking.py      Document -> passages (section-bounded, overlapping)
├── embeddings/      hash, fastembed, OpenAI-compatible
├── store.py         DuckDB schema, BM25 + vector SQL, chapter reassembly
├── indexer.py       incremental indexing and re-embedding
├── search.py        query parsing, RRF fusion, re-rankers
├── qa.py, llm.py    grounded answers (LLM or extractive)
├── goodreads.py     CSV metadata import
├── app.py           the Bookworm facade
├── cli.py           Click + Rich command line
└── mcp.py           MCP server (JSON-RPC over stdio)
```

## Extraction

Every extractor returns a `Document`: title, author, format, page count, and a list of
`Section`s. A section is a run of paragraphs that share a chapter number, chapter title and
heading path (for example `("Maintenance", "Oil")`). Paragraphs carry a page number when the
format has pages.

`SectionBuilder` hides the bookkeeping. Extractors call `start_chapter`, `start_subsection`
and `add_paragraph` as they walk the file; chapter numbers are assigned lazily, so a heading
with nothing under it (a part title, a stray contents entry) does not consume a number.

Where chapters come from:

| Format | Chapter source | Fallback |
|---|---|---|
| EPUB | EPUB 3 `nav` or EPUB 2 NCX table of contents | first heading in each spine document |
| PDF | top-level outline (bookmarks) | one untitled run with page numbers |
| Markdown / HTML | the shallowest heading level used more than once | none |
| TXT | lines like `CHAPTER IV.`, `Chapter 3: The Storm`, `BOOK TWO` | none |

EPUB, HTML and Markdown parsing use only the standard library. PDF and MOBI support are
optional extras so the base install stays small.

## Chunking

`chunk_sections` packs whole paragraphs into passages of about `chunk_tokens` (320 by
default) and never lets a passage cross a section boundary. That rule is what makes every
citation clean: one passage, one chapter, one heading path, one page range.

- Paragraphs longer than the budget are split at sentence boundaries. A sentence that is
  still too long (tables, OCR noise) is split by word count as a last resort.
- Each passage starts with up to `chunk_overlap` tokens (64 by default) repeated from the
  end of the previous one: whole short paragraphs when they fit, otherwise the trailing
  sentences of the last paragraph.
- A runt final passage is folded into its neighbour when the result stays within 125% of
  the budget.
- Each chunk stores `overlap_chars`, so `bookworm read` reassembles a chapter's exact text
  with no duplicated sentences.

Token counts are estimated from word counts (about 4 tokens per 3 words), which is accurate
enough for sizing and avoids shipping a tokenizer.

## Storage

One DuckDB file holds four tables:

- `books`: path, title, author, format, SHA-256, size, mtime, page and chapter counts,
  Goodreads fields.
- `chunks`: text, citation fields, token and term counts, a content hash, and the
  embedding as a `FLOAT[]`.
- `postings`: the BM25 inverted index, one row per (term, chunk) with its term frequency.
- `meta`: schema version, the embedder that built the index, and the chunking signature.

Inserts use DuckDB's `unnest` over parallel arrays, so a whole book's chunks and postings
land in two statements.

## Retrieval

**BM25** runs as a single SQL query: document frequencies come from the postings table,
document lengths from `chunks.n_terms`, with the usual `k1 = 1.2`, `b = 0.75`. Query and
index terms go through the same normaliser: Unicode case folding, accent stripping, a stop
list, and a deliberately light stemmer (plurals and possessives only, since aggressive
stemming hurts precision on literary text).

**Vector search** is an exact scan with `list_dot_product` over L2-normalised vectors, which
equals cosine similarity. DuckDB's vectorised execution keeps this fast for personal-library
scale (hundreds of books, tens of thousands of passages) with no extension to install or
index to maintain.

**Hybrid search** takes the top 40 candidates from each ranking and fuses them with
Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank)`. RRF needs no score calibration between
BM25 and cosine, which is why it is the default.

**Re-rankers** then post-process the fused list:

- `AdjacentDedupe` (always on) drops a passage when a better-ranked passage is its
  immediate neighbour in the same book, since neighbours share overlap text.
- `MMRReranker` (on when `BOOKWORM_MMR_LAMBDA < 1`) applies Maximal Marginal Relevance
  using the stored embeddings, trading a little relevance for variety.

**Phrases** in double quotes become `strpos` filters applied inside both retrievers, so
they must appear verbatim.

## Incremental indexing

For each file the indexer takes the cheapest correct path:

1. Same path, size and mtime: skip without reading the file.
2. Same SHA-256 as the stored book: record the new mtime and skip.
3. New path whose hash matches a book whose old file is gone: treat it as a move and update
   the path. No extraction, no embedding.
4. Otherwise extract and re-chunk, then look up every chunk's content hash among the stored
   chunks and embed only the ones that are new. Editing one chapter re-embeds one chapter.

If the chunking settings change, the next `index` run re-chunks everything (still reusing
vectors for identical passages). If the embedder changes, searches are refused until
`bookworm reindex` re-embeds the library, because vectors from different models are not
comparable.

## Question answering

`ask` retrieves the top passages with hybrid search, then:

- with a chat model, sends them as numbered sources under a system prompt that requires
  inline `[n]` citations and an explicit "not in the sources" when appropriate;
- without one, scores every sentence of the top passages by IDF-weighted coverage of the
  question's terms, keeps the best few in reading order, and cites each.

Both paths return the same `Answer` object with the list of source numbers actually cited.

## MCP server

`bookworm mcp` speaks newline-delimited JSON-RPC 2.0 on stdio and implements `initialize`,
`ping`, `tools/list` and `tools/call` directly, with no SDK dependency. Tool failures (an
unknown book, a missing argument) come back in-band with `isError: true` so the calling model
can recover. The database is opened read-only.
