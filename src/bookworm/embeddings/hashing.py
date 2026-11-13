"""A dependency-free, offline embedder based on feature hashing.

Words and character trigrams are hashed into a fixed number of signed buckets
(the "hashing trick"), weighted sub-linearly and L2-normalised. It needs no model download
and no network, is fully deterministic, and handles typos and word-form variation
("marry", "marrying", "married") better than exact keyword match. It is lexical, not truly
semantic: for paraphrase-level matching, switch to ``fastembed`` or an API model.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from bookworm.embeddings.base import l2_normalize
from bookworm.text import STOPWORDS, char_ngrams, stem, words

WORD_WEIGHT = 1.0
NGRAM_WEIGHT = 0.35


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hash:v1:{self.dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def _embed(self, text: str) -> list[float]:
        features: dict[str, float] = {}
        for word in words(text):
            if word in STOPWORDS:
                continue
            stemmed = stem(word)
            features[f"w:{stemmed}"] = features.get(f"w:{stemmed}", 0.0) + WORD_WEIGHT
            for gram in char_ngrams(stemmed):
                features[f"g:{gram}"] = features.get(f"g:{gram}", 0.0) + NGRAM_WEIGHT
        vector = [0.0] * self.dim
        for feature, weight in features.items():
            index, sign = self._bucket(feature)
            vector[index] += sign * (1.0 + math.log(weight)) if weight >= 1 else sign * weight
        return l2_normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
