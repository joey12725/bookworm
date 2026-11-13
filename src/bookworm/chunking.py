"""Structure-aware chunking.

Chunks never cross a chapter or section boundary, so every passage has one clean citation.
Inside a section, whole paragraphs are packed up to a token budget; paragraphs that are too
long on their own are split at sentence boundaries. Consecutive chunks share a few trailing
sentences of overlap so an idea that straddles a boundary is still retrievable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bookworm.models import Chunk, Section
from bookworm.text import approx_tokens, split_sentences


@dataclass(slots=True, frozen=True)
class ChunkingConfig:
    target_tokens: int = 320
    overlap_tokens: int = 64
    #: A trailing chunk smaller than this fraction of the target is folded into its neighbour.
    min_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.target_tokens < 16:
            raise ValueError("target_tokens must be at least 16")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("overlap_tokens must be >= 0 and smaller than target_tokens")

    @property
    def signature(self) -> str:
        """Stored with the index; a change means existing chunks are stale."""
        return f"v2:{self.target_tokens}:{self.overlap_tokens}:{self.min_fraction}"


@dataclass(slots=True)
class _Unit:
    text: str
    tokens: int
    page: int | None
    paragraph_start: bool
    carried: bool = False  # repeated from the previous chunk as overlap


def _units(section: Section, cfg: ChunkingConfig) -> list[_Unit]:
    """Break a section into packable units: paragraphs, or sentences of long paragraphs."""
    units: list[_Unit] = []
    for para in section.paragraphs:
        tokens = approx_tokens(para.text)
        if tokens <= cfg.target_tokens:
            units.append(_Unit(para.text, tokens, para.page, True))
            continue
        first = True
        for sentence in split_sentences(para.text):
            for piece in _hard_split(sentence, cfg.target_tokens):
                units.append(_Unit(piece, approx_tokens(piece), para.page, first))
                first = False
    return units


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Last resort for run-on 'sentences' (tables, OCR noise): split by word count."""
    if approx_tokens(text) <= max_tokens:
        return [text]
    words = text.split()
    step = max(1, max_tokens * 3 // 4)
    return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]


def _join(units: list[_Unit]) -> str:
    out: list[str] = []
    for unit in units:
        if out and unit.paragraph_start:
            out.append("\n\n")
        elif out:
            out.append(" ")
        out.append(unit.text)
    return "".join(out)


def _overlap(group: list[_Unit], budget: int) -> list[_Unit]:
    """Trailing context to repeat at the start of the next chunk: whole units if they fit,
    otherwise the last sentences of the final paragraph."""
    carried: list[_Unit] = []
    size = 0
    for unit in reversed(group):
        if size + unit.tokens > budget:
            break
        carried.insert(0, unit)
        size += unit.tokens
    if not carried and group:
        sentences = split_sentences(group[-1].text)
        tail: list[str] = []
        for sentence in reversed(sentences[1:]):
            if size + approx_tokens(sentence) > budget:
                break
            tail.insert(0, sentence)
            size += approx_tokens(sentence)
        if tail:
            text = " ".join(tail)
            return [_Unit(text, approx_tokens(text), group[-1].page, True, carried=True)]
    return [
        _Unit(u.text, u.tokens, u.page, u.paragraph_start or i == 0, carried=True)
        for i, u in enumerate(carried)
    ]


def _pack(units: list[_Unit], cfg: ChunkingConfig) -> list[list[_Unit]]:
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    size = 0
    for unit in units:
        has_fresh = any(not u.carried for u in current)
        if has_fresh and size + unit.tokens > cfg.target_tokens:
            groups.append(current)
            current = _overlap(current, cfg.overlap_tokens)
            size = sum(u.tokens for u in current)
            if size + unit.tokens > cfg.target_tokens:
                current, size = [], 0
        current.append(unit)
        size += unit.tokens
    if any(not u.carried for u in current):
        groups.append(current)

    # Fold a runt tail into the previous chunk when it fits comfortably.
    if len(groups) >= 2:
        tail = [u for u in groups[-1] if not u.carried]
        tail_tokens = sum(u.tokens for u in tail)
        prev_tokens = sum(u.tokens for u in groups[-2])
        if (
            tail_tokens < cfg.target_tokens * cfg.min_fraction
            and prev_tokens + tail_tokens <= cfg.target_tokens * 1.25
        ):
            groups[-2] = groups[-2] + tail
            groups.pop()
    return groups


def chunk_sections(sections: Iterable[Section], cfg: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkingConfig()
    chunks: list[Chunk] = []
    for section in sections:
        for group in _pack(_units(section, cfg), cfg):
            text = _join(group)
            n_carried = sum(1 for u in group if u.carried)
            overlap = 0
            if 0 < n_carried < len(group):
                overlap = len(_join(group[: n_carried + 1])) - len(group[n_carried].text)
            pages = [u.page for u in group if u.page is not None]
            chunks.append(
                Chunk(
                    ord=len(chunks),
                    text=text,
                    chapter=section.chapter,
                    chapter_title=section.chapter_title,
                    section=" > ".join(section.path) or None,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    tokens=approx_tokens(text),
                    overlap_chars=overlap,
                )
            )
    return chunks
