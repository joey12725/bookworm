from __future__ import annotations

import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from bookworm import Bookworm, Settings
from bookworm.embeddings import HashingEmbedder

SAMPLES = Path(__file__).resolve().parent.parent / "examples" / "library"


class CountingEmbedder:
    """Wraps the hashing embedder and counts how many texts it embedded."""

    def __init__(self, dim: int = 256) -> None:
        self.inner = HashingEmbedder(dim)
        self.embedded = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A writable copy of the bundled public-domain sample library."""
    dest = tmp_path / "library"
    shutil.copytree(SAMPLES, dest)
    return dest


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "test.duckdb", hash_dim=256)


@pytest.fixture
def embedder() -> CountingEmbedder:
    return CountingEmbedder()


@pytest.fixture
def bw(settings: Settings, embedder: CountingEmbedder, library: Path) -> Iterator[Bookworm]:
    with Bookworm(settings, embedder=embedder) as instance:
        instance.index([library])
        yield instance
