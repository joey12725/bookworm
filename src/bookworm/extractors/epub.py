"""EPUB 2 and 3, read with the standard library (zipfile + ElementTree).

Reading order comes from the OPF spine. Chapter titles come from the table of contents
(EPUB 3 nav document or EPUB 2 NCX), falling back to the first heading in each file.
"""

from __future__ import annotations

import posixpath
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import unquote

from bookworm.errors import ExtractionError
from bookworm.extractors.base import SectionBuilder
from bookworm.extractors.html import blocks_to_sections, parse_html
from bookworm.models import Document
from bookworm.text import normalize_whitespace

NS = {
    "c": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "epub": "http://www.idpf.org/2007/ops",
}
MIN_CHAPTER_CHARS = 200


def _read_xml(zf: zipfile.ZipFile, name: str) -> ET.Element:
    try:
        return ET.fromstring(zf.read(name))
    except (KeyError, ET.ParseError) as exc:
        raise ExtractionError(f"Malformed EPUB: cannot read {name}") from exc


def _resolve(base_dir: str, href: str) -> str:
    return posixpath.normpath(posixpath.join(base_dir, unquote(href.split("#", 1)[0])))


def _toc_titles(zf: zipfile.ZipFile, opf: ET.Element, opf_dir: str) -> dict[str, str]:
    """Map content-file path to the first table-of-contents title that points into it."""
    manifest = opf.find("opf:manifest", NS)
    if manifest is None:
        return {}
    titles: dict[str, str] = {}

    nav = next(
        (i for i in manifest if "nav" in (i.get("properties") or "").split()),
        None,
    )
    if nav is not None and nav.get("href"):
        nav_path = _resolve(opf_dir, nav.get("href", ""))
        root = _read_xml(zf, nav_path)
        nav_dir = posixpath.dirname(nav_path)
        for nav_el in root.iter(f"{{{NS['xhtml']}}}nav"):
            if nav_el.get(f"{{{NS['epub']}}}type") not in (None, "toc"):
                continue
            for a in nav_el.iter(f"{{{NS['xhtml']}}}a"):
                text = normalize_whitespace("".join(a.itertext()))
                if a.get("href") and text:
                    titles.setdefault(_resolve(nav_dir, a.get("href", "")), text)
        if titles:
            return titles

    spine = opf.find("opf:spine", NS)
    toc_id = spine.get("toc") if spine is not None else None
    ncx_item = next((i for i in manifest if i.get("id") == toc_id), None)
    if ncx_item is not None and ncx_item.get("href"):
        ncx_path = _resolve(opf_dir, ncx_item.get("href", ""))
        root = _read_xml(zf, ncx_path)
        ncx_dir = posixpath.dirname(ncx_path)
        for point in root.iter(f"{{{NS['ncx']}}}navPoint"):
            label = point.find("ncx:navLabel/ncx:text", NS)
            content = point.find("ncx:content", NS)
            if label is not None and label.text and content is not None and content.get("src"):
                titles.setdefault(
                    _resolve(ncx_dir, content.get("src", "")), normalize_whitespace(label.text)
                )
    return titles


def extract_epub(path: Path) -> Document:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Not a valid EPUB (zip) file: {path.name}") from exc

    with zf:
        container = _read_xml(zf, "META-INF/container.xml")
        rootfile = container.find(".//c:rootfile", NS)
        if rootfile is None or not rootfile.get("full-path"):
            raise ExtractionError("Malformed EPUB: container.xml has no rootfile")
        opf_path = rootfile.get("full-path", "")
        opf = _read_xml(zf, opf_path)
        opf_dir = posixpath.dirname(opf_path)

        title_el = opf.find(".//dc:title", NS)
        author_el = opf.find(".//dc:creator", NS)
        title = normalize_whitespace(title_el.text or "") if title_el is not None else ""
        author = normalize_whitespace(author_el.text or "") if author_el is not None else ""

        manifest = {item.get("id"): item for item in opf.iterfind("opf:manifest/opf:item", NS)}
        toc = _toc_titles(zf, opf, opf_dir)
        builder = SectionBuilder()

        for itemref in opf.iterfind("opf:spine/opf:itemref", NS):
            if itemref.get("linear") == "no":
                continue
            item = manifest.get(itemref.get("idref"))
            if item is None or "nav" in (item.get("properties") or "").split():
                continue
            if "html" not in (item.get("media-type") or ""):
                continue
            doc_path = _resolve(opf_dir, item.get("href", ""))
            try:
                markup = zf.read(doc_path).decode("utf-8", errors="replace")
            except KeyError:
                continue
            blocks, _ = parse_html(markup)
            body_chars = sum(len(b.text) for b in blocks if b.heading_level is None)
            if body_chars == 0:
                continue

            first_heading = next((b for b in blocks if b.heading_level is not None), None)
            chapter_title = toc.get(doc_path) or (first_heading.text if first_heading else None)
            is_chapter = doc_path in toc or first_heading is not None
            if is_chapter and (body_chars >= MIN_CHAPTER_CHARS or doc_path in toc):
                builder.start_chapter(chapter_title)
                if first_heading is not None:
                    blocks = [b for b in blocks if b is not first_heading]
            blocks_to_sections(builder, blocks, chapter_level=None)

    return Document(
        title=title or path.stem,
        author=author or None,
        format="epub",
        sections=builder.finish(),
        pages=None,
    )
