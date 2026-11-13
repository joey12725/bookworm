"""Plain-text books, with first-class handling of Project Gutenberg files.

Chapters are detected from heading lines such as ``CHAPTER IV.``, ``Chapter 3: The Storm``
or ``BOOK TWO``. Gutenberg license boilerplate and illustration markers are stripped.
"""

from __future__ import annotations

import re
from pathlib import Path

from bookworm.extractors.base import SectionBuilder, read_text_file, split_paragraphs
from bookworm.models import Document

_NUMBER = (
    r"(?:[IVXLCDM]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|first|second|"
    r"third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final)"
)
HEADING_RE = re.compile(
    rf"^(?:chapter|book|part|letter|section|canto|stave|act)\s+{_NUMBER}\b\.?(?:[\s.:\-—–]+(?P<title>\S.*))?$",
    re.IGNORECASE,
)
_START_RE = re.compile(r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
_END_RE = re.compile(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
_META_RE = re.compile(r"^(Title|Author):\s*(.+)$", re.M)
_ITALIC_RE = re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)")
MAX_HEADING_CHARS = 80


def strip_bracketed(text: str, opener: str = "[Illustration") -> str:
    """Remove ``[Illustration ...]`` blocks, including nested brackets and blank lines."""
    out: list[str] = []
    i = 0
    while (start := text.find(opener, i)) != -1:
        out.append(text[i:start])
        depth = 0
        j = start
        while j < len(text):
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        i = j + 1
    out.append(text[i:])
    return "".join(out)


def split_gutenberg(text: str) -> tuple[str, dict[str, str]]:
    """Return the body between the Gutenberg START/END markers, plus header metadata."""
    meta: dict[str, str] = {}
    start = _START_RE.search(text)
    header = text[: start.start()] if start else "\n".join(text.split("\n")[:40])
    for key, value in _META_RE.findall(header):
        meta.setdefault(key.lower(), value.strip())
    if start:
        text = text[start.end() :]
        if end := _END_RE.search(text):
            text = text[: end.start()]
    elif meta:
        # Plain header block without markers: drop the metadata lines themselves.
        text = _META_RE.sub("", text)
    return text, meta


_ROMAN_RE = re.compile(r"^[IVXLCDM]+[.:]?$")


def tidy_heading(heading: str) -> str:
    """'CHAPTER IV. THE STORM.' -> 'Chapter IV. The Storm'; 'CHAPTER I. The Start' keeps
    its mixed-case title."""
    words = heading.strip().rstrip(".").split()
    label, rest = words[0].capitalize(), words[1:]
    shout = " ".join(w for w in rest if not _ROMAN_RE.match(w)).isupper()
    tidied = [w if _ROMAN_RE.match(w) or not shout else w.capitalize() for w in rest]
    return " ".join([label, *tidied])


def match_heading(paragraph: str) -> tuple[bool, str | None]:
    if len(paragraph) > MAX_HEADING_CHARS:
        return False, None
    if not HEADING_RE.match(paragraph.strip()):
        return False, None
    return True, tidy_heading(paragraph)


def extract_text(path: Path) -> Document:
    body, meta = split_gutenberg(read_text_file(path))
    body = _ITALIC_RE.sub(r"\1", strip_bracketed(body))

    builder = SectionBuilder()
    for paragraph in split_paragraphs(body):
        is_heading, title = match_heading(paragraph)
        if is_heading:
            builder.start_chapter(title)
        else:
            builder.add_paragraph(paragraph)

    return Document(
        title=meta.get("title") or path.stem.replace("-", " ").replace("_", " ").title(),
        author=meta.get("author"),
        format="txt",
        sections=builder.finish(),
    )
