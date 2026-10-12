from __future__ import annotations

from collections.abc import Iterable

from bookworm_core.models import ChunkRecord, TextFragment
from bookworm_core.utils import normalize_whitespace


MAX_CHUNK_TOKENS = 2000  # text-embedding-3-small limit is 8191 tokens; real text ~3-4 tokens/word


def estimate_tokens(text: str) -> int:
    # Fast approximation: sentence-transformer uses subword tokens; words is close enough for chunk sizing.
    return max(1, len(text.split()))


def _split_paragraphs(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").split("\n")
    paragraphs: list[str] = []
    current: list[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(normalize_whitespace(" ".join(current)))
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(normalize_whitespace(" ".join(current)))

    return [p for p in paragraphs if p]


def _split_long_paragraph(text: str, max_tokens: int) -> list[str]:
    """Split a single oversized paragraph into pieces by sentences, then by words."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        t = estimate_tokens(sent)
        if t > max_tokens:
            # Sentence itself is too long, split by words
            if current:
                pieces.append(" ".join(current))
                current, current_tokens = [], 0
            words = sent.split()
            for i in range(0, len(words), max_tokens):
                pieces.append(" ".join(words[i:i + max_tokens]))
        elif current_tokens + t > max_tokens and current:
            pieces.append(" ".join(current))
            current, current_tokens = [sent], t
        else:
            current.append(sent)
            current_tokens += t
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_fragments(
    fragments: Iterable[TextFragment],
    *,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[ChunkRecord]:
    all_chunks: list[ChunkRecord] = []
    chunk_index = 0

    for fragment in fragments:
        raw_paragraphs = _split_paragraphs(fragment.content)
        if not raw_paragraphs:
            continue

        # Split any oversized paragraphs to stay within embedding model limits
        paragraphs: list[str] = []
        for p in raw_paragraphs:
            t = estimate_tokens(p)
            if t > MAX_CHUNK_TOKENS:
                paragraphs.extend(_split_long_paragraph(p, MAX_CHUNK_TOKENS))
            else:
                paragraphs.append(p)

        paragraph_tokens = [estimate_tokens(p) for p in paragraphs]
        start = 0

        while start < len(paragraphs):
            end = start
            running_tokens = 0

            while end < len(paragraphs) and (
                running_tokens < target_tokens or end == start
            ):
                running_tokens += paragraph_tokens[end]
                end += 1

            chunk_text = "\n\n".join(paragraphs[start:end]).strip()
            if chunk_text:
                all_chunks.append(
                    ChunkRecord(
                        content=chunk_text,
                        chapter_title=fragment.chapter_title,
                        chapter_number=fragment.chapter_number,
                        page_number=fragment.page_number,
                        chunk_index=chunk_index,
                        token_count=estimate_tokens(chunk_text),
                    )
                )
                chunk_index += 1

            if end >= len(paragraphs):
                break

            if overlap_tokens <= 0:
                start = end
                continue

            overlap_running = 0
            new_start = end
            while new_start > start and overlap_running < overlap_tokens:
                new_start -= 1
                overlap_running += paragraph_tokens[new_start]

            if new_start == start:
                start = end
            else:
                start = new_start

    return all_chunks
