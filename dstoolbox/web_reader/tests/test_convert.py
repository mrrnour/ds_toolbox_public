"""Tests for convert.py — written before implementation."""

import json
import logging
import os

import pytest

from dstoolbox.web_reader import convert


def make_args(**kwargs):
    defaults = dict(
        input_path=None,
        output_dir=None,
        output_path=None,
        min_length=50,
        overwrite=True,
        dry_run=False,
        verbose=False,
        use_docling=False,
    )
    return convert.ConvertConfig(**{**defaults, **kwargs})


@pytest.fixture
def logger():
    return logging.getLogger("test")


def make_crawl_summary(tmp_path, pages):
    """pages = list of dicts like {url, html, depth, scraped_at, status, ext, bytes}."""
    html_dir = tmp_path / "html"
    html_dir.mkdir(exist_ok=True)
    summary = tmp_path / "crawling_summary.jsonl"
    records = []
    for i, p in enumerate(pages):
        ext = p.get("ext", ".html")
        file_path = str(html_dir / f"page_{i}{ext}")
        if p.get("write_html", True):
            if "bytes" in p:
                with open(file_path, "wb") as f:
                    f.write(p["bytes"])
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(p["html"])
        records.append({
            "id": p["url"],
            "status": p.get("status", "ok"),
            "data": {
                "file_path": file_path,
                "depth": p.get("depth", 0),
                "status_code": 200,
                "content_length": len(p.get("html", p.get("bytes", b""))),
                "scraped_at": p.get("scraped_at", "2026-04-16T12:00:00"),
            },
            "error": None,
        })
    with open(summary, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return summary, html_dir, records


def test_happy_path_produces_md_with_frontmatter(tmp_path, logger):
    html = """
    <html><head><title>T</title><script>var x=1;</script><style>.a{}</style></head>
    <body>
      <nav>NAV</nav>
      <header>HEADER</header>
      <h1>Hello</h1>
      <p>This is a paragraph with enough text to pass min length threshold easily.</p>
      <ul><li>one</li><li>two</li></ul>
      <footer>FOOTER</footer>
    </body></html>
    """
    summary, html_dir, _ = make_crawl_summary(tmp_path, [
        {"url": "https://example.com/a", "html": html, "depth": 0,
         "scraped_at": "2026-04-16T12:00:00"},
    ])
    out_dir = tmp_path / "md"
    out = tmp_path / "conversion_summary.jsonl"

    args = make_args(
        input_path=str(summary),
        output_dir=str(out_dir), output_path=str(out),
        min_length=20,
    )
    convert.run(args, logger)

    records = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "ok"

    md_path = records[0]["data"]["md_path"]
    assert os.path.exists(md_path)
    md = open(md_path, encoding="utf-8").read()
    assert md.startswith("---\n")
    assert "url: https://example.com/a" in md
    assert "scraped_at: 2026-04-16T12:00:00" in md
    assert "depth: 0" in md
    assert "Hello" in md
    assert "var x=1" not in md
    assert "NAV" not in md
    assert "HEADER" not in md
    assert "FOOTER" not in md


def test_short_content_skipped_not_halted(tmp_path, logger):
    summary, html_dir, _ = make_crawl_summary(tmp_path, [
        {"url": "https://example.com/short",
         "html": "<html><body><p>hi</p></body></html>", "depth": 0},
    ])
    args = make_args(
        input_path=str(summary),
        output_dir=str(tmp_path / "md"),
        output_path=str(tmp_path / "out.jsonl"),
        min_length=500,
    )
    convert.run(args, logger)
    records = [json.loads(ln) for ln in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert records[0]["status"] == "skipped"
    assert records[0]["reason"] == "below_min_length"


def test_missing_html_file_is_failed_not_halted(tmp_path, logger):
    summary, html_dir, _ = make_crawl_summary(tmp_path, [
        {"url": "https://example.com/good",
         "html": "<html><body><p>" + ("x" * 300) + "</p></body></html>",
         "depth": 0},
        {"url": "https://example.com/missing", "html": "ignored",
         "depth": 0, "write_html": False},
    ])
    args = make_args(
        input_path=str(summary),
        output_dir=str(tmp_path / "md"),
        output_path=str(tmp_path / "out.jsonl"),
        min_length=50,
    )
    convert.run(args, logger)
    records = [json.loads(ln) for ln in (tmp_path / "out.jsonl").read_text().splitlines()]
    statuses = [r["status"] for r in records]
    assert "ok" in statuses
    assert "failed" in statuses


def test_skipped_crawl_records_are_passed_through_as_skipped(tmp_path, logger):
    summary, html_dir, _ = make_crawl_summary(tmp_path, [
        {"url": "https://example.com/pdf",
         "html": "%PDF fake", "depth": 0, "status": "skipped"},
    ])
    args = make_args(
        input_path=str(summary),
        output_dir=str(tmp_path / "md"),
        output_path=str(tmp_path / "out.jsonl"),
        min_length=20,
    )
    convert.run(args, logger)
    records = [json.loads(ln) for ln in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert records[0]["status"] == "skipped"


def test_dry_run_writes_nothing(tmp_path, logger):
    summary, html_dir, _ = make_crawl_summary(tmp_path, [
        {"url": "https://example.com/a",
         "html": "<html><body>" + ("x" * 300) + "</body></html>", "depth": 0},
    ])
    out_dir = tmp_path / "md"
    out = tmp_path / "out.jsonl"
    args = make_args(
        input_path=str(summary),
        output_dir=str(out_dir), output_path=str(out),
        dry_run=True,
    )
    convert.run(args, logger)
    assert not out.exists()
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_missing_input_exits(tmp_path, logger):
    args = make_args(
        input_path=str(tmp_path / "nope.jsonl"),
        output_dir=str(tmp_path / "md"),
        output_path=str(tmp_path / "out.jsonl"),
    )
    with pytest.raises(SystemExit):
        convert.run(args, logger)


def _minimal_pdf_bytes(text: str = "Hello PDF world") -> bytes:
    """Build a tiny single-page PDF from scratch so tests don't need a real file."""
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject, DictionaryObject, FloatObject, NameObject, NumberObject, TextStringObject,
    )
    import io
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    page = writer.pages[0]
    content = f"BT /F1 12 Tf 50 100 Td ({text}) Tj ET".encode()
    from pypdf.generic import ByteStringObject, StreamObject
    stream = StreamObject()
    stream._data = content
    stream.update({NameObject("/Length"): NumberObject(len(content))})
    page[NameObject("/Contents")] = stream
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    resources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    page[NameObject("/Resources")] = resources
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_converted_to_markdown(tmp_path, logger):
    pdf_bytes = _minimal_pdf_bytes("Hello from PDF")
    summary, html_dir, _ = make_crawl_summary(tmp_path, [
        {"url": "https://example.com/doc.pdf", "ext": ".pdf", "bytes": pdf_bytes,
         "depth": 0, "scraped_at": "2026-04-16T12:00:00"},
    ])
    out_dir = tmp_path / "md"
    out = tmp_path / "out.jsonl"

    args = make_args(
        input_path=str(summary),
        output_dir=str(out_dir), output_path=str(out),
        min_length=1,
    )
    convert.run(args, logger)

    records = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert records[0]["status"] == "ok"
    md_path = records[0]["data"]["md_path"]
    assert md_path.endswith(".md")
    md = open(md_path, encoding="utf-8").read()
    assert md.startswith("---\n")
    assert "url: https://example.com/doc.pdf" in md
    assert "Hello from PDF" in md


def test_strip_boilerplate_removes_tags():
    from bs4 import BeautifulSoup
    html = "<html><body><script>x</script><nav>n</nav><p>keep</p></body></html>"
    soup = convert.strip_boilerplate(BeautifulSoup(html, "html.parser"))
    text = str(soup)
    assert "keep" in text
    assert "script" not in text
    assert "nav" not in text
