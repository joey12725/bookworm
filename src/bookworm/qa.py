"""Grounded question answering with numbered citations.

With a chat model configured, retrieved passages go to the model as numbered sources and it
must cite them inline as [1], [2]. Without one, Bookworm answers extractively: it picks the
sentences from the top passages that best cover the question and cites each one.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from bookworm.llm import ChatModel
from bookworm.models import Answer, Hit
from bookworm.text import split_sentences, terms

SYSTEM_PROMPT = (
    "You answer questions about the user's books using only the numbered source passages "
    "provided. Cite every claim inline with the source number in square brackets, like [2]. "
    "Quote short phrases where the wording matters. If the sources do not contain the "
    "answer, say so plainly instead of guessing."
)
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def format_sources(hits: Sequence[Hit]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        blocks.append(f"[{n}] {hit.citation.format()}\n{hit.text}")
    return "\n\n".join(blocks)


def cited_numbers(text: str, n_sources: int) -> list[int]:
    found: list[int] = []
    for group in _CITE_RE.findall(text):
        for raw in group.split(","):
            n = int(raw)
            if 1 <= n <= n_sources and n not in found:
                found.append(n)
    return found


def extractive_answer(question: str, hits: Sequence[Hit], max_sentences: int = 3) -> str:
    """Pick the sentences that best cover the question, citing each.

    Query terms are weighted by how rare they are among the candidate sentences, so
    "supreme excellence" outweighs "war" in a book about war.
    """
    wanted = set(terms(question))
    candidates: list[tuple[int, int, str, set[str]]] = []
    for n, hit in enumerate(hits, start=1):
        for position, sentence in enumerate(split_sentences(hit.text)):
            sentence_terms = set(terms(sentence)) & wanted
            if sentence_terms:
                candidates.append((n, position, sentence, sentence_terms))
    if not candidates:
        return "I couldn't find anything in your library that answers that."

    df = Counter(t for *_, found in candidates for t in found)
    idf = {t: math.log(1 + len(candidates) / df[t]) for t in df}
    total = sum(idf.get(t, math.log(1 + len(candidates))) for t in wanted)

    scored = [
        (sum(idf[t] for t in found) / total + 0.1 / n, n, position, sentence)
        for n, position, sentence, found in candidates
    ]
    scored.sort(key=lambda s: -s[0])
    floor = scored[0][0] * 0.6
    best = [s for s in scored[:max_sentences] if s[0] >= floor]
    best.sort(key=lambda s: (s[1], s[2]))  # reading order within and across sources
    return " ".join(f"{sentence} [{n}]" for _, n, _, sentence in best)


def answer(question: str, hits: list[Hit], chat: ChatModel | None) -> Answer:
    if not hits:
        return Answer(
            question, "No passages in the library matched that question.", [], [], "extractive"
        )
    if chat is None:
        text = extractive_answer(question, hits)
        return Answer(question, text, hits, cited_numbers(text, len(hits)), "extractive")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}",
        },
    ]
    text = chat.complete(messages)
    return Answer(question, text, hits, cited_numbers(text, len(hits)), "llm")
