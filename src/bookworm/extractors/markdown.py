"""Markdown books and notes. Headings drive structure; YAML-style front matter supplies
title and author."""

from __future__ import annotations

import re
from pathlib import Path

from bookworm.extractors.base import SectionBuilder, read_text_file
from bookworm.extractors.html import Block, blocks_to_sections, chapter_heading_level
from bookworm.models import Document
from bookworm.text import normalize_whitespace

_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")
_INLINE = [
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), ""),  # images
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # links
    (re.compile(r"`([^`]+)`"), r"\1"),
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),
    (re.compile(r"(?<![\w*])([*_])(?!\s)(.+?)(?<!\s)\1(?![\w*])"), r"\2"),
    (re.compile(r"<[^>]+>"), ""),
]
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def strip_inline(text: str) -> str:
    for pattern, repl in _INLINE:
        text = pattern.sub(repl, text)
    return text


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, text[match.end() :]


def markdown_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        if buffer:
            joined = normalize_whitespace(strip_inline(" ".join(buffer)))
            if joined:
                blocks.append(Block(joined))
            buffer.clear()

    for line in text.split("\n"):
        if _FENCE_RE.match(line.strip()):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            buffer.append(line)
            continue
        if heading := _HEADING_RE.match(line):
            flush()
            blocks.append(
                Block(strip_inline(heading.group(2)), heading_level=len(heading.group(1)))
            )
        elif not line.strip() or _LIST_RE.match(line):
            flush()
            if line.strip():
                buffer.append(_LIST_RE.sub("", line))
        elif line.lstrip().startswith(">"):
            buffer.append(line.lstrip().lstrip(">"))
        else:
            buffer.append(line)
    flush()
    return blocks


def extract_markdown(path: Path) -> Document:
    meta, body = parse_front_matter(read_text_file(path))
    blocks = markdown_blocks(body)
    title = meta.get("title")
    level = chapter_heading_level(blocks)
    if level is not None:
        above = [b for b in blocks if b.heading_level is not None and b.heading_level < level]
        if len(above) == 1:
            title = title or above[0].text
            blocks = [b for b in blocks if b is not above[0]]
    builder = SectionBuilder()
    blocks_to_sections(builder, blocks, chapter_level=level)
    return Document(
        title=title or path.stem.replace("-", " ").replace("_", " ").title(),
        author=meta.get("author"),
        format="md",
        sections=builder.finish(),
    )
