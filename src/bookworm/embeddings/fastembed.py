"""Local ONNX embeddings via fastembed (``pip install 'bookworm[fastembed]'``).

Real semantic search with no API key and no PyTorch. The model (about 130 MB for the
default) is downloaded once on first use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bookworm.embeddings.base import l2_normalize
from bookworm.errors import MissingDependencyError

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedEmbedder:
    def __init__(self, model: str | None = None, batch_size: int = 32) -> None:
        self.model = model or DEFAULT_MODEL
        self.batch_size = batch_size
        self._model: Any = None

    @property
    def name(self) -> str:
        return f"fastembed:{self.model}"

    def _load(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise MissingDependencyError("The fastembed embedder", "fastembed") from exc
            self._model = TextEmbedding(model_name=self.model)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        return [
            l2_normalize(v.tolist()) for v in model.embed(list(texts), batch_size=self.batch_size)
        ]

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        embed = getattr(model, "query_embed", model.embed)
        return l2_normalize(next(iter(embed([text]))).tolist())
