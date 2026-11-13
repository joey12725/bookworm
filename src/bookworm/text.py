"""Text utilities shared by the chunker, the BM25 index and the hashing embedder."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterator

_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s+[\"'“‘(\[_]*[A-Z0-9])")
_ABBREVIATIONS = ("Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Sr.", "Jr.", "Prof.", "vs.", "e.g.", "i.e.")
_ABBREV_TOKEN = "\u0000"

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves
    """.split()
)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def fold(text: str) -> str:
    """Lowercase, strip accents and unify curly quotes so 'Café' matches 'cafe'."""
    text = text.replace("’", "'").replace("‘", "'")
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def stem(word: str) -> str:
    """A deliberately light English stemmer: possessives and regular plurals only.

    Aggressive stemmers hurt precision on literary text ("general" vs "generous"), so this
    only folds the forms people almost always mean to match.
    """
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def words(text: str) -> list[str]:
    return _WORD_RE.findall(fold(text))


def terms(text: str) -> list[str]:
    """Index terms for BM25: folded, stop-word filtered, lightly stemmed."""
    return [stem(w) for w in words(text) if (w not in STOPWORDS and len(w) > 1) or w.isdigit()]


def approx_tokens(text: str) -> int:
    """Rough sub-word token count (about 4 tokens per 3 English words)."""
    n = len(text.split())
    return max(1, math.ceil(n * 4 / 3)) if n else 0


def split_sentences(text: str) -> list[str]:
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr[:-1] + _ABBREV_TOKEN)
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(protected):
        parts.append(protected[start : match.end()])
        start = match.end()
    parts.append(protected[start:])
    return [p.replace(_ABBREV_TOKEN, ".").strip() for p in parts if p.strip()]


def char_ngrams(word: str, n: int = 3) -> Iterator[str]:
    padded = f"<{word}>"
    if len(padded) <= n:
        yield padded
        return
    for i in range(len(padded) - n + 1):
        yield padded[i : i + n]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snippet(text: str, query: str, width: int = 360) -> str:
    """A short excerpt of ``text`` starting at the sentence that best matches ``query``."""
    flat = normalize_whitespace(text)
    wanted = set(terms(query))
    sentences = split_sentences(flat)
    best_index = 0
    best_score = 0
    for i, sentence in enumerate(sentences):
        score = len(wanted & set(terms(sentence)))
        if score > best_score:
            best_index, best_score = i, score
    excerpt = " ".join(sentences[best_index:]) if sentences else flat
    prefix = "… " if best_index > 0 else ""
    if len(excerpt) + len(prefix) <= width:
        return prefix + excerpt
    cut = excerpt[: width - len(prefix) - 2].rsplit(" ", 1)[0]
    return f"{prefix}{cut} …"
