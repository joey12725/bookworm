"""Retrieval: BM25, vector, and hybrid search with rank fusion and diversity re-ranking.

Hybrid search runs both retrievers and fuses their rankings with Reciprocal Rank Fusion
(RRF). Re-rankers then post-process the fused candidates: by default, adjacent chunks of
the same passage are collapsed so the top results are not overlapping slices of one page,
and Maximal Marginal Relevance (MMR) can be switched on for stronger diversity.
Re-rankers are pluggable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from bookworm.embeddings import Embedder
from bookworm.models import Hit, SearchMode
from bookworm.store import Store
from bookworm.text import terms

_PHRASE_RE = re.compile(r'"([^"]+)"|“([^”]+)”')


@dataclass(slots=True, frozen=True)
class ParsedQuery:
    text: str
    terms: tuple[str, ...]
    phrases: tuple[str, ...]


def parse_query(query: str) -> ParsedQuery:
    """Split out ``"quoted phrases"``, which must appear verbatim in results."""
    phrases = tuple(a or b for a, b in _PHRASE_RE.findall(query))
    plain = _PHRASE_RE.sub(lambda m: m.group(1) or m.group(2), query)
    return ParsedQuery(text=plain.strip(), terms=tuple(terms(plain)), phrases=phrases)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], *, k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[int, float]]:
    """Fuse ranked id lists: score(d) = sum_i w_i / (k + rank_i(d))."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Reranker(Protocol):
    """Reorders candidate hits. Receives more candidates than it should return."""

    def rerank(self, query: str, hits: list[Hit], store: Store, limit: int) -> list[Hit]: ...


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


@dataclass(slots=True)
class AdjacentDedupe:
    """Drop a hit when a better-ranked hit is its neighbouring chunk in the same section.

    Neighbouring chunks share overlap text, so showing both mostly repeats the passage."""

    radius: int = 1

    def rerank(self, query: str, hits: list[Hit], store: Store, limit: int) -> list[Hit]:
        kept: list[Hit] = []
        for hit in hits:
            if not any(self._overlaps(k, hit) for k in kept):
                kept.append(hit)
        return kept

    def _overlaps(self, a: Hit, b: Hit) -> bool:
        return (
            a.book_id == b.book_id
            and abs(a.ord - b.ord) <= self.radius
            and a.citation.chapter == b.citation.chapter
            and a.citation.section == b.citation.section
        )


@dataclass(slots=True)
class MMRReranker:
    """Maximal Marginal Relevance: trade relevance against similarity to results already
    picked. ``lambda_=1`` is pure relevance; lower values favour diversity."""

    lambda_: float = 0.75

    def rerank(self, query: str, hits: list[Hit], store: Store, limit: int) -> list[Hit]:
        if len(hits) <= 1:
            return hits[:limit]
        vectors = store.embeddings_for([h.chunk_id for h in hits])
        top = max(h.score for h in hits)
        relevance = {h.chunk_id: (h.score / top if top > 0 else 0.0) for h in hits}

        selected: list[Hit] = []
        remaining = list(hits)
        while remaining and len(selected) < limit:

            def mmr(hit: Hit) -> float:
                vec = vectors.get(hit.chunk_id)
                redundancy = 0.0
                if vec is not None:
                    redundancy = max(
                        (_dot(vec, vectors[s.chunk_id]) for s in selected if s.chunk_id in vectors),
                        default=0.0,
                    )
                return self.lambda_ * relevance[hit.chunk_id] - (1 - self.lambda_) * redundancy

            best = max(remaining, key=mmr)
            selected.append(best)
            remaining.remove(best)
        return selected


@dataclass(slots=True)
class Retriever:
    store: Store
    embedder: Embedder
    rrf_k: int = 60
    candidates: int = 40
    rerankers: list[Reranker] = field(default_factory=list)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        mode: SearchMode = "hybrid",
        book_ids: Sequence[int] | None = None,
    ) -> list[Hit]:
        parsed = parse_query(query)
        if not parsed.text:
            return []
        pool = max(self.candidates, limit * 4)

        if mode == "keyword":
            scored = self.store.search_bm25(
                parsed.terms, limit=pool, book_ids=book_ids, phrases=parsed.phrases
            )
            return self._rerank(parsed.text, self.store.hits(scored), limit)

        if mode == "vector":
            self.store.bind_embedder(self.embedder.name)
            scored = self.store.search_vector(
                self.embedder.embed_query(parsed.text),
                limit=pool,
                book_ids=book_ids,
                phrases=parsed.phrases,
            )
            return self._rerank(parsed.text, self.store.hits(scored), limit)

        self.store.bind_embedder(self.embedder.name)
        keyword = self.store.search_bm25(
            parsed.terms, limit=pool, book_ids=book_ids, phrases=parsed.phrases
        )
        vector = self.store.search_vector(
            self.embedder.embed_query(parsed.text),
            limit=pool,
            book_ids=book_ids,
            phrases=parsed.phrases,
        )
        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in keyword], [cid for cid, _ in vector]], k=self.rrf_k
        )
        return self._rerank(parsed.text, self.store.hits(fused[:pool]), limit)

    def _rerank(self, query: str, hits: list[Hit], limit: int) -> list[Hit]:
        for reranker in self.rerankers:
            hits = reranker.rerank(query, hits, self.store, limit)
        return hits[:limit]
