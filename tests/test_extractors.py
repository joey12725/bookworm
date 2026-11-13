from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from bookworm.errors import ExtractionError, UnsupportedFormatError
from bookworm.extractors import extract, register, supported_extensions
from bookworm.models import Document, Section

from .conftest import SAMPLES
from .helpers import build_epub, build_pdf

GUTENBERG = """The Project Gutenberg eBook of A Tiny Tale

Title: A Tiny Tale
Author: Ada Example

*** START OF THE PROJECT GUTENBERG EBOOK A TINY TALE ***

Some front matter.

CHAPTER I.
The Beginning

It was a _dark_ and stormy night.
The rain fell.

[Illustration:

“A picture”

[_Copyright 1900_]]

The wind howled.

CHAPTER II.

Morning came.

*** END OF THE PROJECT GUTENBERG EBOOK A TINY TALE ***
Licence text that must not be indexed.
"""


def test_plaintext_gutenberg(tmp_path: Path) -> None:
    path = tmp_path / "tiny.txt"
    path.write_text(GUTENBERG, encoding="utf-8")
    doc = extract(path)
    assert (doc.title, doc.author, doc.format) == ("A Tiny Tale", "Ada Example", "txt")
    assert doc.chapter_count == 2
    front, first, second = doc.sections
    assert front.chapter is None and front.text == "Some front matter."
    assert first.chapter_title == "Chapter I. The Beginning"
    assert first.text == "It was a dark and stormy night. The rain fell.\n\nThe wind howled."
    assert second.chapter_title == "Chapter II"
    full = "\n".join(s.text for s in doc.sections)
    assert "Illustration" not in full and "Licence" not in full


def test_plaintext_without_headings_uses_filename(tmp_path: Path) -> None:
    path = tmp_path / "field-notes.txt"
    path.write_bytes("Caf\xe9 notes.\n\nPart of the reason.".encode("cp1252"))
    doc = extract(path)
    assert doc.title == "Field Notes"
    assert doc.chapter_count == 0  # "Part of the reason" is prose, not a heading
    assert doc.sections[0].paragraphs[0].text == "Café notes."


def test_markdown_structure(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        '---\ntitle: Field Guide\nauthor: "B. Writer"\n---\n'
        "# Field Guide\n\nIntro with a [link](https://example.com).\n\n"
        "## Birds\n\nSome **bold** birds.\n\n### Owls\n\n- nocturnal\n- quiet\n\n"
        "## Trees\n\n```\ncode stays\n```\n\n![img](x.png)Oaks.\n",
        encoding="utf-8",
    )
    doc = extract(path)
    assert (doc.title, doc.author) == ("Field Guide", "B. Writer")
    assert [(s.chapter, s.chapter_title, s.path) for s in doc.sections] == [
        (None, None, ()),
        (1, "Birds", ()),
        (1, "Birds", ("Owls",)),
        (2, "Trees", ()),
    ]
    assert doc.sections[0].text == "Intro with a link."
    assert doc.sections[1].text == "Some bold birds."
    assert doc.sections[2].text == "nocturnal\n\nquiet"
    assert doc.sections[3].text == "code stays\n\nOaks."


def test_html(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Essays</title><style>p{}</style></head><body>"
        "<h2>One</h2><p>First &amp; best.</p><script>x()</script>"
        "<h2>Two</h2><p>Second<br>line.</p></body></html>",
        encoding="utf-8",
    )
    doc = extract(path)
    assert doc.title == "Essays"
    assert [(s.chapter_title, s.text) for s in doc.sections] == [
        ("One", "First & best."),
        ("Two", "Second\n\nline."),
    ]


@pytest.mark.parametrize("nav", [True, False], ids=["epub3-nav", "epub2-ncx"])
def test_epub_uses_toc_titles(tmp_path: Path, nav: bool) -> None:
    path = build_epub(
        tmp_path / "book.epub",
        title="Short Stories",
        author="C. Author",
        chapters=[("The Lighthouse", ["Waves.", "Gulls."]), ("The Harbour", ["Boats."])],
        nav=nav,
    )
    doc = extract(path)
    assert (doc.title, doc.author, doc.format) == ("Short Stories", "C. Author", "epub")
    assert [(s.chapter, s.chapter_title, s.text) for s in doc.sections] == [
        (1, "The Lighthouse", "Waves.\n\nGulls."),
        (2, "The Harbour", "Boats."),
    ]


def test_bad_epub_raises_extraction_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.epub"
    path.write_bytes(b"not a zip")
    with pytest.raises(ExtractionError):
        extract(path)


def test_pdf_pages_and_outline(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    path = build_pdf(
        tmp_path / "manual.pdf",
        title="Repair Manual",
        author="D. Mechanic",
        pages=[
            ["Preface text."],
            ["Engines need oil.", "", "Check it weekly."],
            ["Brakes need pads."],
        ],
        outline={1: "Engines", 2: "Brakes"},
    )
    doc = extract(path)
    assert (doc.title, doc.author, doc.pages) == ("Repair Manual", "D. Mechanic", 3)
    by_chapter = {s.chapter_title: s for s in doc.sections}
    assert set(by_chapter) == {None, "Engines", "Brakes"}
    assert {p.page for p in by_chapter["Engines"].paragraphs} == {2}
    assert "oil" in by_chapter["Engines"].text
    assert by_chapter["Brakes"].paragraphs[0].page == 3


def test_mobi_adapter_with_fake_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unpacked = tmp_path / "unpacked"
    (unpacked / "mobi7").mkdir(parents=True)
    (unpacked / "mobi7" / "book.html").write_text(
        "<h1>Chapter 1</h1><p>Alpha text.</p><h1>Chapter 2</h1><p>Beta text.</p>"
    )
    (unpacked / "content.opf").write_text(
        "<dc:title>Kindle Book</dc:title><dc:creator>E. K.</dc:creator>"
    )
    fake = types.ModuleType("mobi")
    fake.extract = lambda path: (str(unpacked), str(unpacked / "mobi7" / "book.html"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mobi", fake)
    book = tmp_path / "book.mobi"
    book.write_bytes(b"MOBI")
    doc = extract(book)
    assert (doc.title, doc.author, doc.chapter_count) == ("Kindle Book", "E. K.", 2)
    assert not unpacked.exists()  # temp dir cleaned up


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.docx"
    path.write_text("x")
    with pytest.raises(UnsupportedFormatError):
        extract(path)


def test_register_custom_extractor(tmp_path: Path) -> None:
    @register(".fake")
    def extract_fake(path: Path) -> Document:
        return Document("Fake", None, "fake", [Section(None, None)])

    assert ".fake" in supported_extensions()
    path = tmp_path / "x.fake"
    path.write_text("")
    assert extract(path).title == "Fake"


@pytest.mark.parametrize(
    ("name", "title", "chapters"),
    [
        ("pride-and-prejudice.txt", "Pride and Prejudice", 3),
        ("alice-in-wonderland.md", "Alice's Adventures in Wonderland", 2),
        ("the-art-of-war.epub", "The Art of War", 3),
    ],
)
def test_bundled_samples(name: str, title: str, chapters: int) -> None:
    doc = extract(SAMPLES / name)
    assert doc.title == title
    assert doc.chapter_count == chapters
