from __future__ import annotations

import html
import re
from pathlib import Path

import shutil
import tempfile

import ebooklib
import fitz
import mobi
from ebooklib import epub

from bookworm_core.errors import ExtractionError
from bookworm_core.models import TextFragment
from bookworm_core.utils import normalize_whitespace

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
HEADING_RE = re.compile(r"<h[1-3][^>]*>(.*?)</h[1-3]>", re.IGNORECASE | re.DOTALL)


def extract_from_pdf(path: Path) -> tuple[list[TextFragment], dict[str, str | int | None]]:
    try:
        doc = fitz.open(path)
    except Exception as exc:  # pragma: no cover - backend specific failures
        raise ExtractionError(f"Failed to open PDF: {path}") from exc

    fragments: list[TextFragment] = []
    try:
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            text = page.get_text("text")
            cleaned = text.strip()
            if not cleaned:
                continue
            fragments.append(TextFragment(content=cleaned, page_number=page_idx + 1))

        title = (doc.metadata or {}).get("title") if doc.metadata else None
        author = (doc.metadata or {}).get("author") if doc.metadata else None

        return fragments, {
            "title": normalize_whitespace(title) if title else path.stem,
            "author": normalize_whitespace(author) if author else None,
            "total_pages": doc.page_count,
        }
    finally:
        doc.close()


def _html_to_text(content: bytes) -> tuple[str, str | None]:
    decoded = content.decode("utf-8", errors="ignore")
    heading_match = HEADING_RE.search(decoded)
    heading = None
    if heading_match:
        heading = html.unescape(TAG_RE.sub(" ", heading_match.group(1))).strip()
        heading = normalize_whitespace(heading) if heading else None

    without_script = SCRIPT_STYLE_RE.sub(" ", decoded)
    text = TAG_RE.sub(" ", without_script)
    text = html.unescape(text)
    text = normalize_whitespace(text)
    return text, heading


def extract_from_epub(path: Path) -> tuple[list[TextFragment], dict[str, str | int | None]]:
    try:
        book = epub.read_epub(str(path))
    except Exception as exc:  # pragma: no cover - backend specific failures
        raise ExtractionError(f"Failed to open EPUB: {path}") from exc

    title_meta = book.get_metadata("DC", "title")
    author_meta = book.get_metadata("DC", "creator")

    title = title_meta[0][0] if title_meta else path.stem
    author = author_meta[0][0] if author_meta else None

    fragments: list[TextFragment] = []
    chapter_number = 0

    # Respect spine order so search results line up with reading order.
    for spine_entry in book.spine:
        if not spine_entry:
            continue

        item_id = spine_entry[0]
        item = book.get_item_with_id(item_id)
        if not item or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        try:
            content = item.get_content()
        except (KeyError, Exception):
            continue

        text, heading = _html_to_text(content)
        if not text:
            continue

        chapter_number += 1
        fragments.append(
            TextFragment(
                content=text,
                chapter_title=heading,
                chapter_number=chapter_number,
                page_number=None,
            )
        )

    # Fallback: if spine yielded nothing, try iterating all document items directly
    if not fragments:
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            try:
                content = item.get_content()
            except (KeyError, Exception):
                continue

            text, heading = _html_to_text(content)
            if not text:
                continue

            chapter_number += 1
            fragments.append(
                TextFragment(
                    content=text,
                    chapter_title=heading,
                    chapter_number=chapter_number,
                    page_number=None,
                )
            )

    return fragments, {
        "title": normalize_whitespace(title) if title else path.stem,
        "author": normalize_whitespace(author) if author else None,
        "total_pages": chapter_number if chapter_number > 0 else None,
    }


def extract_from_mobi(path: Path) -> tuple[list[TextFragment], dict[str, str | int | None]]:
    """Extract text from .mobi files by unpacking to HTML via kindleunpack."""
    tmpdir = None
    try:
        tmpdir, _ = mobi.extract(str(path))
    except Exception as exc:
        raise ExtractionError(f"Failed to open MOBI: {path}") from exc

    try:
        # Find the main HTML file (mobi7/book.html or similar)
        html_files = sorted(Path(tmpdir).rglob("*.html"))
        if not html_files:
            html_files = sorted(Path(tmpdir).rglob("*.htm"))
        if not html_files:
            raise ExtractionError(f"No HTML content found in MOBI: {path}")

        # Read the largest HTML file (the main content)
        main_html = max(html_files, key=lambda f: f.stat().st_size)
        raw = main_html.read_bytes()

        # Try to extract title/author from OPF if present
        opf_files = list(Path(tmpdir).rglob("*.opf"))
        title = path.stem
        author = None
        if opf_files:
            opf_text = opf_files[0].read_text(errors="ignore")
            import re as _re
            t_match = _re.search(r"<dc:title[^>]*>(.*?)</dc:title>", opf_text, _re.IGNORECASE | _re.DOTALL)
            a_match = _re.search(r"<dc:creator[^>]*>(.*?)</dc:creator>", opf_text, _re.IGNORECASE | _re.DOTALL)
            if t_match:
                title = html.unescape(TAG_RE.sub("", t_match.group(1))).strip()
            if a_match:
                author = html.unescape(TAG_RE.sub("", a_match.group(1))).strip()

        # Split on heading tags to get pseudo-chapters
        chapter_splits = re.split(r"(<h[1-3][^>]*>.*?</h[1-3]>)", raw.decode("utf-8", errors="ignore"), flags=re.IGNORECASE | re.DOTALL)

        fragments: list[TextFragment] = []
        chapter_number = 0

        # If no headings found, treat the whole thing as one fragment
        if len(chapter_splits) <= 1:
            text, heading = _html_to_text(raw)
            if text:
                fragments.append(TextFragment(content=text, chapter_title=heading, chapter_number=1, page_number=None))
        else:
            current_heading = None
            for i, part in enumerate(chapter_splits):
                if re.match(r"<h[1-3]", part, re.IGNORECASE):
                    current_heading = html.unescape(TAG_RE.sub(" ", part)).strip()
                    current_heading = normalize_whitespace(current_heading) if current_heading else None
                else:
                    text, _ = _html_to_text(part.encode("utf-8", errors="ignore"))
                    if not text or len(text) < 20:
                        continue
                    chapter_number += 1
                    fragments.append(TextFragment(content=text, chapter_title=current_heading, chapter_number=chapter_number, page_number=None))
                    current_heading = None

        return fragments, {
            "title": normalize_whitespace(title) if title else path.stem,
            "author": normalize_whitespace(author) if author else None,
            "total_pages": chapter_number if chapter_number > 0 else None,
        }
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def extract_book(path: Path) -> tuple[list[TextFragment], dict[str, str | int | None]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_from_pdf(path)
    if suffix == ".epub":
        return extract_from_epub(path)
    if suffix == ".mobi":
        return extract_from_mobi(path)
    raise ExtractionError(f"Unsupported file type: {path.suffix}")
