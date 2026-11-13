"""Kindle MOBI/AZW via the ``mobi`` package (``pip install 'bookworm[mobi]'``).

The book is unpacked to HTML and handed to the HTML pipeline.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from bookworm.errors import ExtractionError, MissingDependencyError
from bookworm.extractors.base import SectionBuilder
from bookworm.extractors.html import blocks_to_sections, chapter_heading_level, parse_html
from bookworm.models import Document
from bookworm.text import normalize_whitespace

_OPF_FIELD = r"<dc:{0}[^>]*>(.*?)</dc:{0}>"


def extract_mobi(path: Path) -> Document:
    try:
        import mobi
    except ImportError as exc:
        raise MissingDependencyError("MOBI support", "mobi") from exc

    try:
        tmpdir, _ = mobi.extract(str(path))
    except Exception as exc:
        raise ExtractionError(f"Could not unpack MOBI {path.name}: {exc}") from exc

    try:
        root = Path(tmpdir)
        pages = sorted(root.rglob("*.html")) or sorted(root.rglob("*.htm"))
        if not pages:
            raise ExtractionError(f"No HTML content found in {path.name}")
        main = max(pages, key=lambda p: p.stat().st_size)
        blocks, html_title = parse_html(main.read_text(encoding="utf-8", errors="replace"))

        meta: dict[str, str] = {}
        if opf := next(iter(root.rglob("*.opf")), None):
            opf_text = opf.read_text(encoding="utf-8", errors="replace")
            for name in ("title", "creator"):
                if m := re.search(_OPF_FIELD.format(name), opf_text, re.I | re.S):
                    meta[name] = normalize_whitespace(re.sub(r"<[^>]+>", "", m.group(1)))

        builder = SectionBuilder()
        blocks_to_sections(builder, blocks, chapter_level=chapter_heading_level(blocks))
        return Document(
            title=meta.get("title") or html_title or path.stem,
            author=meta.get("creator"),
            format=path.suffix.lower().lstrip("."),
            sections=builder.finish(),
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
