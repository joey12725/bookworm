"""HTML/XHTML to structured text, using only the standard library.

Shared by the EPUB and MOBI extractors and used directly for ``.html`` files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from bookworm.extractors.base import SectionBuilder, read_text_file
from bookworm.models import Document
from bookworm.text import normalize_whitespace

_BLOCK_TAGS = {
    "p", "div", "section", "article", "blockquote", "li", "dd", "dt", "pre",
    "tr", "table", "ul", "ol", "br", "hr", "figure", "figcaption", "aside", "header", "footer",
}  # fmt: skip
_SKIP_TAGS = {"script", "style", "head", "title", "svg", "math", "nav"}
_HEADING_RE = re.compile(r"^h([1-6])$")


@dataclass(slots=True)
class Block:
    text: str
    heading_level: int | None = None  # 1-6 for <h1>-<h6>, None for body text


class _BlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.title: str | None = None
        self._buffer: list[str] = []
        self._skip_depth = 0
        self._heading: int | None = None
        self._in_title = False
        self._title_buffer: list[str] = []

    def _flush(self) -> None:
        text = normalize_whitespace("".join(self._buffer))
        self._buffer = []
        if text:
            self.blocks.append(Block(text=text, heading_level=self._heading))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if match := _HEADING_RE.match(tag):
            self._flush()
            self._heading = int(match.group(1))
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"br", "hr"}:
            self._flush()
        else:
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            self.title = normalize_whitespace("".join(self._title_buffer)) or None
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if _HEADING_RE.match(tag):
            self._flush()
            self._heading = None
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_buffer.append(data)
        if not self._skip_depth:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def parse_html(markup: str) -> tuple[list[Block], str | None]:
    """Return the text blocks of an HTML document and its ``<title>``."""
    parser = _BlockParser()
    parser.feed(markup)
    parser.close()
    return parser.blocks, parser.title


def blocks_to_sections(
    builder: SectionBuilder,
    blocks: list[Block],
    *,
    chapter_level: int | None = None,
    page: int | None = None,
) -> None:
    """Feed blocks into a SectionBuilder. Headings at ``chapter_level`` start chapters;
    deeper headings become sub-sections. With ``chapter_level=None`` the caller has already
    opened the chapter and every heading is treated as a sub-section."""
    for block in blocks:
        if block.heading_level is None:
            builder.add_paragraph(block.text, page=page)
        elif chapter_level is not None and block.heading_level <= chapter_level:
            builder.start_chapter(block.text)
        else:
            depth = block.heading_level - (chapter_level or 1)
            builder.start_subsection(block.text, depth=max(1, depth))


def chapter_heading_level(blocks: list[Block]) -> int | None:
    """Pick the heading level that marks chapters: the shallowest level used more than once,
    skipping a lone top-level book title."""
    counts: dict[int, int] = {}
    for b in blocks:
        if b.heading_level is not None:
            counts[b.heading_level] = counts.get(b.heading_level, 0) + 1
    if not counts:
        return None
    for level in sorted(counts):
        if counts[level] > 1:
            return level
    return max(counts)


def extract_html(path: Path) -> Document:
    blocks, title = parse_html(read_text_file(path))
    level = chapter_heading_level(blocks)
    builder = SectionBuilder()
    book_title = title
    if level is not None and level > 1:
        top = [b for b in blocks if b.heading_level is not None and b.heading_level < level]
        if len(top) == 1 and not book_title:
            book_title = top[0].text
        blocks = [b for b in blocks if b.heading_level is None or b.heading_level >= level]
    blocks_to_sections(builder, blocks, chapter_level=level)
    return Document(
        title=book_title or path.stem,
        author=None,
        format="html",
        sections=builder.finish(),
    )
