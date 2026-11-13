from __future__ import annotations

import pytest

from bookworm import Bookworm
from bookworm.errors import BookNotFoundError, IndexMismatchError
from bookworm.models import Citation, Hit
from bookworm.search import MMRReranker, parse_query, reciprocal_rank_fusion
from bookworm.store import Store


def test_parse_query_extracts_phrases() -> None:
    q = parse_query('"truth universally" wife')
    assert q.phrases == ("truth universally",)
    assert q.text == "truth universally wife"
    assert q.terms == ("truth", "universally", "wife")


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    ids = [i for i, _ in fused]
    assert ids[:2] == [1, 3]
    assert set(ids) == {1, 2, 3, 4}


def test_keyword_search_finds_famous_line(bw: Bookworm) -> None:
    (hit, *_) = bw.search("truth universally acknowledged", mode="keyword", limit=3)
    assert hit.citation.book == "Pride and Prejudice"
    assert hit.citation.chapter == 1
    assert "single man in possession of a good fortune" in hit.text


def test_phrase_filter_requires_exact_text(bw: Bookworm) -> None:
    hits = bw.search('"Curiouser and curiouser"', mode="keyword", limit=10)
    assert hits and all("curiouser and curiouser" in h.text.lower() for h in hits)
    assert bw.search('"curiouser and pride"', mode="keyword") == []


def test_vector_search_handles_word_variants(bw: Bookworm) -> None:
    (hit, *_) = bw.search("generals winning without fighting battles", mode="vector")
    assert hit.citation.book == "The Art of War"


@pytest.mark.parametrize(
    ("query", "book"),
    [
        ("rabbit with a pocket watch", "Alice's Adventures in Wonderland"),
        ("supreme excellence breaking the enemy's resistance without fighting", "The Art of War"),
        ("Mr. Darcy refused to dance at the assembly", "Pride and Prejudice"),
    ],
)
def test_hybrid_search_picks_the_right_book(bw: Bookworm, query: str, book: str) -> None:
    hits = bw.search(query, limit=3)
    assert hits[0].citation.book == book
    assert hits[0].score > 0


def test_book_filter(bw: Bookworm) -> None:
    hits = bw.search("war", book="alice", limit=5)
    assert hits and {h.citation.book for h in hits} == {"Alice's Adventures in Wonderland"}
    with pytest.raises(BookNotFoundError):
        bw.search("war", book="Moby Dick")


def test_hits_carry_citations(bw: Bookworm) -> None:
    (hit,) = bw.search("Curiouser and curiouser", limit=1)
    assert (
        hit.citation.format() == "Alice's Adventures in Wonderland, Chapter II. The Pool of Tears"
    )
    data = hit.to_dict()
    assert data["citation"]["label"] == hit.citation.format()


def test_citation_format_with_pages() -> None:
    c = Citation("Manual", None, 2, None, "Oil", 10, 12)
    assert c.format() == "Manual, ch. 2, Oil, pp. 10-12"
    assert Citation("Manual", None, None, None, None, 7, 7).format() == "Manual, p. 7"


def test_mmr_prefers_diverse_results(bw: Bookworm) -> None:
    store = bw.store
    ids = [cid for cid, _ in store.search_bm25(["alice"], limit=6)]
    vectors = store.embeddings_for(ids)
    first = ids[0]
    # Make a fake candidate list where the 2nd item is a duplicate of the 1st.
    hits = store.hits([(first, 1.0), (first, 0.99), *[(i, 0.5) for i in ids[1:4]]])
    hits[1] = Hit(-1, hits[0].book_id, 0, hits[0].text, 0.99, hits[0].citation)
    vectors[-1] = vectors[first]
    store.embeddings_for = lambda chunk_ids: vectors  # type: ignore[method-assign]
    picked = MMRReranker(lambda_=0.5).rerank("q", hits, store, limit=2)
    assert picked[0].chunk_id == first
    assert picked[1].chunk_id != -1


def test_embedder_mismatch_is_refused(bw: Bookworm, settings) -> None:
    from bookworm.embeddings import HashingEmbedder

    other = Bookworm(settings, embedder=HashingEmbedder(dim=64), store=bw.store)
    with pytest.raises(IndexMismatchError, match="reindex"):
        other.search("anything", mode="vector")
    assert other.search("alice", mode="keyword")  # keyword search needs no embedder


def test_empty_store(tmp_path) -> None:
    with Store(tmp_path / "empty.duckdb") as store:
        assert store.search_bm25(["anything"], limit=5) == []
        assert store.hits([]) == []
        assert store.stats()["books"] == 0


def test_adjacent_chunks_are_collapsed(bw: Bookworm) -> None:
    from bookworm.search import AdjacentDedupe

    ids = [cid for cid, _ in bw.store.search_bm25(["alice"], limit=20)]
    hits = bw.store.hits([(cid, 1.0) for cid in ids])
    kept = AdjacentDedupe().rerank("alice", hits, bw.store, limit=20)
    pairs = {(h.book_id, h.citation.chapter, h.ord) for h in kept}
    assert all((b, c, o + 1) not in pairs for b, c, o in pairs)
    assert kept[0] == hits[0]
