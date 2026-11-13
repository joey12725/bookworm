"""Builders for tiny, valid test books (EPUB and PDF) so no binary fixtures are needed."""

from __future__ import annotations

import zipfile
from html import escape
from pathlib import Path

CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""


def build_epub(
    path: Path,
    *,
    title: str,
    author: str,
    chapters: list[tuple[str, list[str]]],
    nav: bool = True,
) -> Path:
    """Write a minimal EPUB 3 (``nav=True``) or EPUB 2 with an NCX table of contents."""
    manifest = []
    spine = []
    toc_entries = []
    files: dict[str, str] = {}
    for i, (chapter_title, paragraphs) in enumerate(chapters, start=1):
        name = f"chapter{i:02d}.xhtml"
        body = "\n".join(f"<p>{escape(p)}</p>" for p in paragraphs)
        files[name] = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
            f"<title>{escape(chapter_title)}</title></head><body>"
            f"<h1>{escape(chapter_title)}</h1>\n{body}</body></html>"
        )
        manifest.append(f'<item id="c{i}" href="{name}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="c{i}"/>')
        toc_entries.append((name, chapter_title))

    if nav:
        links = "".join(f'<li><a href="{n}">{escape(t)}</a></li>' for n, t in toc_entries)
        files["nav.xhtml"] = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
            f'<head><title>Contents</title></head><body><nav epub:type="toc"><ol>{links}</ol></nav>'
            "</body></html>"
        )
        manifest.append(
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
        )
        spine_attr = ""
    else:
        points = "".join(
            f'<navPoint id="p{i}" playOrder="{i}"><navLabel><text>{escape(t)}</text></navLabel>'
            f'<content src="{n}"/></navPoint>'
            for i, (n, t) in enumerate(toc_entries, start=1)
        )
        files["toc.ncx"] = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
            f"<navMap>{points}</navMap></ncx>"
        )
        manifest.append('<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
        spine_attr = ' toc="ncx"'

    opf = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f'<dc:identifier id="id">bookworm-{escape(title.lower().replace(" ", "-"))}</dc:identifier>'
        f"<dc:title>{escape(title)}</dc:title><dc:creator>{escape(author)}</dc:creator>"
        "<dc:language>en</dc:language></metadata>"
        f"<manifest>{''.join(manifest)}</manifest>"
        f"<spine{spine_attr}>{''.join(spine)}</spine></package>"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", CONTAINER, zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/content.opf", opf, zipfile.ZIP_DEFLATED)
        for name, content in files.items():
            zf.writestr(f"OEBPS/{name}", content, zipfile.ZIP_DEFLATED)
    return path


def build_pdf(
    path: Path, *, title: str, author: str, pages: list[list[str]], outline: dict[int, str]
) -> Path:
    """Write a text PDF with one list of lines per page and top-level bookmarks.

    ``outline`` maps a 0-based page index to a bookmark title.
    """
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 720 Td"]
        for line in lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({safe}) Tj T*")
        ops.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
    for index, label in sorted(outline.items()):
        writer.add_outline_item(title=label, page_number=index)
    writer.add_metadata({"/Title": title, "/Author": author})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        writer.write(f)
    return path
