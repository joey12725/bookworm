# Extending Bookworm

The extension points are small protocols. None of them need changes to Bookworm itself.

## Add a file format

Write a function that turns a path into a `Document` and register it for one or more
extensions. `SectionBuilder` takes care of chapter numbering and section paths.

```python
from pathlib import Path

from bookworm.extractors import register
from bookworm.extractors.base import SectionBuilder, read_text_file, split_paragraphs
from bookworm.models import Document


@register(".org")
def extract_org(path: Path) -> Document:
    builder = SectionBuilder()
    for block in split_paragraphs(read_text_file(path)):
        if block.startswith("* "):
            builder.start_chapter(block[2:])
        elif block.startswith("** "):
            builder.start_subsection(block[3:])
        else:
            builder.add_paragraph(block)
    return Document(title=path.stem, author=None, format="org", sections=builder.finish())
```

Import the module before indexing (for example in your own entry point) and `.org` files
are picked up by `discover` and `index` like any other format.

## Add an embedder

Anything with a `name`, `embed_documents` and `embed_query` is an embedder. Return
L2-normalised vectors (use `bookworm.embeddings.l2_normalize`), because vector search scores
with a dot product.

```python
from bookworm import Bookworm, Settings
from bookworm.embeddings import l2_normalize


class MyEmbedder:
    name = "my-model:v1"  # stored with the index; change it when vectors change

    def embed_documents(self, texts):
        return [l2_normalize(v) for v in my_model.encode(list(texts))]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


bw = Bookworm(Settings(db_path="books.duckdb"), embedder=MyEmbedder())
```

## Add a re-ranker

A re-ranker receives more candidates than it should return and gives back a reordered,
possibly shorter, list. A cross-encoder or an LLM judge slots in here.

```python
from bookworm.search import Retriever


class PreferRecentChapters:
    def rerank(self, query, hits, store, limit):
        return sorted(hits, key=lambda h: (-(h.citation.chapter or 0), -h.score))


retriever = Retriever(bw.store, bw.embedder, rerankers=[PreferRecentChapters()])
hits = retriever.search("the storm", limit=5)
```

## Add a chat model

Implement `name` and `complete(messages) -> str` and pass it as `chat=`:

```python
bw = Bookworm(settings, chat=MyChatModel())
print(bw.ask("Who owns the lighthouse?").text)
```

## Running the checks

```bash
uv run pytest
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run mypy
```

`tests/helpers.py` can build a minimal EPUB or text PDF in a few lines, which makes format
tests self-contained.
