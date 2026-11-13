"""The embedder interface."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into vectors.

    ``name`` identifies the vector space (provider and model). The index records it, and a
    different name at query time is refused, since vectors from different models are not
    comparable.
    """

    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return [float(x) for x in vector]
    return [float(x) / norm for x in vector]
