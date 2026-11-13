from __future__ import annotations

import pytest

from bookworm.chunking import ChunkingConfig, chunk_sections
from bookworm.models import Paragraph, Section
from bookworm.text import approx_tokens


def sentence(i: int) -> str:
    return f"Sentence number {i} talks about lighthouses and the sea at some length today."


def section(
    chapter: int | None, n_paragraphs: int, sentences_each: int = 3, page: int | None = None
) -> Section:
    paragraphs = [
        Paragraph(" ".join(sentence(p * 10 + s) for s in range(sentences_each)), page=page)
        for p in range(n_paragraphs)
    ]
    return Section(chapter, f"Chapter {chapter}" if chapter else None, (), paragraphs)


def reassemble(chunks: list) -> list[str]:
    text = chunks[0].text + " " + " ".join(c.text[c.overlap_chars :] for c in chunks[1:])
    return text.split()


def test_chunks_respect_budget_and_section_boundaries() -> None:
    cfg = ChunkingConfig(target_tokens=80, overlap_tokens=20)
    sections = [section(1, 6), section(2, 6)]
    chunks = chunk_sections(sections, cfg)
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    assert all(c.tokens <= cfg.target_tokens * 1.25 for c in chunks)
    for chapter in (1, 2):
        mine = [c for c in chunks if c.chapter == chapter]
        assert all(c.chapter_title == f"Chapter {chapter}" for c in mine)
        source = " ".join(p.text for p in sections[chapter - 1].paragraphs).split()
        assert reassemble(mine) == source


def test_overlap_repeats_trailing_context() -> None:
    cfg = ChunkingConfig(target_tokens=80, overlap_tokens=30)
    chunks = chunk_sections([section(1, 8)], cfg)
    overlapping = [c for c in chunks[1:] if c.overlap_chars]
    assert overlapping, "expected at least one chunk to carry overlap"
    for chunk in overlapping:
        previous = chunks[chunk.ord - 1]
        carried = chunk.text[: chunk.overlap_chars].strip()
        assert previous.text.endswith(carried)
        assert approx_tokens(carried) <= cfg.overlap_tokens


def test_long_paragraph_is_split_at_sentences() -> None:
    cfg = ChunkingConfig(target_tokens=60, overlap_tokens=0)
    chunks = chunk_sections([section(1, 1, sentences_each=20)], cfg)
    assert len(chunks) > 1
    assert all(c.text.rstrip().endswith(".") for c in chunks)


def test_run_on_text_is_hard_split() -> None:
    cfg = ChunkingConfig(target_tokens=40, overlap_tokens=0)
    blob = Section(1, None, (), [Paragraph("word " * 500)])
    chunks = chunk_sections([blob], cfg)
    assert all(c.tokens <= 40 for c in chunks)
    assert sum(len(c.text.split()) for c in chunks) == 500


def test_small_tail_is_folded_into_previous_chunk() -> None:
    cfg = ChunkingConfig(target_tokens=100, overlap_tokens=0)
    paras = [Paragraph("alpha " * 70), Paragraph("omega " * 10)]
    chunks = chunk_sections([Section(1, None, (), paras)], cfg)
    assert len(chunks) == 1


def test_page_ranges_and_section_paths() -> None:
    sec = Section(
        2, "Engines", ("Maintenance", "Oil"), [Paragraph("Check oil.", 4), Paragraph("Top up.", 5)]
    )
    (chunk,) = chunk_sections([sec])
    assert (chunk.page_start, chunk.page_end) == (4, 5)
    assert chunk.section == "Maintenance > Oil"


@pytest.mark.parametrize(("target", "overlap"), [(8, 0), (100, 100), (100, -1)])
def test_invalid_config(target: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(target_tokens=target, overlap_tokens=overlap)


def test_signature_changes_with_settings() -> None:
    assert ChunkingConfig(300, 50).signature != ChunkingConfig(320, 50).signature
