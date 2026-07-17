"""Tests for scraper.py."""

import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dstoolbox.web_reader import scraper


def make_args(**kwargs):
    """Build an argparse-namespace-like object with all run() fields defaulted."""
    defaults = {
        "urls_file": None,
        "fetcher": "requests",
        "timeout_s": 10,
        "timeout_ms": 10_000,
        "max_workers": 2,
        "max_retries": 0,
        "delay": 0.0,
        "headless": True,
        "network_idle": True,
        # output
        "output": None,
        "html_dir": None,
        # pipeline
        "limit": 0,
        "overwrite": False,
        "list_only": False,
        "verbose": False,
    }
    return type("Args", (), {**defaults, **kwargs})()


def fake_response(
    status=200,
    body=b"<html><body>hi</body></html>",
    content_type="text/html; charset=utf-8",
):
    r = MagicMock()
    r.status_code = status
    r.content = body
    r.headers = {"Content-Type": content_type}
    r.raise_for_status = MagicMock()
    if status >= 400:
        import requests as _rq

        r.raise_for_status.side_effect = _rq.HTTPError(f"{status} error")
    return r


@pytest.fixture
def urls_file(tmp_path):
    p = tmp_path / "urls.txt"
    p.write_text(
        "https://example.com/a\n"
        "##https://comment.example.com/skip\n"
        "\n"
        "https://example.com/b \n"
    )
    return p


def _mock_requests_session(monkeypatch, side_effect):
    fake_session = MagicMock()
    fake_session.get.side_effect = side_effect
    fake_session.headers = {}
    monkeypatch.setattr(scraper.requests, "Session", lambda: fake_session)
    return fake_session


# === items_from_file ===

def test_items_from_file_strips_comments_and_blanks(urls_file):
    items, _ = scraper.items_from_file(Path(urls_file))
    urls = [it["url"] for it in items]
    assert urls == ["https://example.com/a", "https://example.com/b"]


def test_items_from_file_missing_exits(tmp_path):
    with pytest.raises(SystemExit):
        scraper.items_from_file(tmp_path / "nope.txt")


# === slug_for ===

def test_slug_for_deterministic():
    s1 = scraper.slug_for("https://example.com/a")
    s2 = scraper.slug_for("https://example.com/a")
    assert s1 == s2
    assert s1.endswith(".html")
    assert len(s1) <= 60


def test_slug_for_uses_ext_arg():
    assert scraper.slug_for("https://example.com/file.pdf", ext=".pdf").endswith(".pdf")


# === ext_for_response ===

def test_ext_for_response_picks_html():
    assert scraper.ext_for_response("https://x/y", "text/html; charset=utf-8") == ".html"


def test_ext_for_response_falls_back_to_url_path():
    assert scraper.ext_for_response("https://x/file.pdf", "") == ".pdf"


def test_ext_for_response_unknown_is_bin():
    assert scraper.ext_for_response("https://x/y", "application/x-weird") == ".bin"


# === backend dispatch ===

def test_fetchers_dispatch_table():
    assert scraper.FETCHERS["requests"] is scraper.make_requests_fetcher
    assert scraper.FETCHERS["stealthy"] is scraper.make_stealthy_fetcher
    assert set(scraper.FETCHERS) == {"requests", "stealthy"}


# === run() with requests backend (mocked) ===

def test_happy_path_requests_writes_html_and_jsonl(tmp_path, urls_file, monkeypatch):
    html_dir = tmp_path / "html"
    out = tmp_path / "crawling_summary.jsonl"

    def fake_get(url, timeout):
        return fake_response(body=f"<html><body>page {url}</body></html>".encode())

    _mock_requests_session(monkeypatch, fake_get)

    args = make_args(
        urls_file=str(urls_file), html_dir=str(html_dir), output=str(out),
        fetcher="requests", max_workers=2,
    )
    rc = scraper.run(args)
    assert rc == 0
    assert out.exists()

    records = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(records) == 2
    for r in records:
        assert r["status"] == "ok"
        assert r["data"]["status_code"] == 200
        # convert.py contract:
        assert r["data"]["depth"] == 0
        assert "scraped_at" in r["data"]
        assert os.path.exists(r["data"]["file_path"])
        assert r["data"]["file_path"].endswith(".html")


def test_pdf_is_saved_with_pdf_extension(tmp_path, monkeypatch):
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.com/file.pdf\n")
    out = tmp_path / "out.jsonl"
    pdf_bytes = b"%PDF-1.4 fake pdf bytes"

    def fake_get(url, timeout):
        return fake_response(status=200, body=pdf_bytes, content_type="application/pdf")

    _mock_requests_session(monkeypatch, fake_get)

    args = make_args(
        urls_file=str(urls), html_dir=str(tmp_path / "html"), output=str(out),
        fetcher="requests", max_workers=1,
    )
    scraper.run(args)

    records = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "ok"
    fp = records[0]["data"]["file_path"]
    assert fp.endswith(".pdf")
    assert os.path.exists(fp)
    with open(fp, "rb") as f:
        assert f.read() == pdf_bytes


def test_empty_body_is_skipped(tmp_path, monkeypatch):
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.com/empty\n")
    out = tmp_path / "out.jsonl"

    def fake_get(url, timeout):
        return fake_response(status=200, body=b"")

    _mock_requests_session(monkeypatch, fake_get)

    args = make_args(
        urls_file=str(urls), html_dir=str(tmp_path / "html"), output=str(out),
        fetcher="requests", max_workers=1,
    )
    scraper.run(args)

    records = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert records[0]["status"] == "skipped"
    assert "empty" in records[0]["reason"].lower()


def test_http_error_is_failed_and_run_continues(tmp_path, monkeypatch):
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.com/broken\nhttps://example.com/good\n")
    out = tmp_path / "out.jsonl"

    def fake_get(url, timeout):
        if "broken" in url:
            return fake_response(status=500, body=b"server error")
        return fake_response(body=b"<html><body>good</body></html>")

    _mock_requests_session(monkeypatch, fake_get)

    args = make_args(
        urls_file=str(urls), html_dir=str(tmp_path / "html"), output=str(out),
        fetcher="requests", max_workers=1,
    )
    scraper.run(args)

    records = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(records) == 2
    statuses = {r["id"]: r["status"] for r in records}
    assert statuses["https://example.com/broken"] == "failed"
    assert statuses["https://example.com/good"] == "ok"

    report = tmp_path / "report.txt"
    assert report.exists()
    assert "broken" in report.read_text()


def test_list_only_writes_nothing(tmp_path, urls_file, monkeypatch):
    out = tmp_path / "out.jsonl"
    fake_session = MagicMock()
    monkeypatch.setattr(scraper.requests, "Session", lambda: fake_session)

    args = make_args(
        urls_file=str(urls_file), html_dir=str(tmp_path / "html"), output=str(out),
        fetcher="requests", list_only=True,
    )
    scraper.run(args)
    assert not out.exists()
    fake_session.get.assert_not_called()


def test_missing_input_exits(tmp_path):
    args = make_args(
        urls_file=str(tmp_path / "nope.txt"),
        html_dir=str(tmp_path / "html"),
        output=str(tmp_path / "out.jsonl"),
        fetcher="requests",
    )
    with pytest.raises(SystemExit):
        scraper.run(args)


# === run() with stealthy backend (mocked) ===

def _patch_stealthy(monkeypatch, page):
    fake_fetcher = MagicMock()
    fake_fetcher.fetch.return_value = page
    fake_module = MagicMock()
    fake_module.StealthyFetcher = fake_fetcher
    monkeypatch.setitem(sys.modules, "scrapling.fetchers", fake_module)
    return fake_fetcher


def test_run_stealthy_backend_mocked(tmp_path, monkeypatch):
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.com/sx\n")
    out = tmp_path / "out.jsonl"

    page = MagicMock()
    page.status = 200
    page.html_content = "<html><body>stealthy page</body></html>"
    fake_fetcher = _patch_stealthy(monkeypatch, page)

    args = make_args(
        urls_file=str(urls), html_dir=str(tmp_path / "html"), output=str(out),
        fetcher="stealthy", max_workers=1,
    )
    rc = scraper.run(args)
    assert rc == 0

    records = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "ok"
    assert records[0]["data"]["status_code"] == 200
    fake_fetcher.fetch.assert_called_once()


def test_stealthy_clamps_max_workers_to_one(tmp_path, monkeypatch, caplog):
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.com/sx\n")
    out = tmp_path / "out.jsonl"

    page = MagicMock(status=200, html_content="<html>hi</html>")
    _patch_stealthy(monkeypatch, page)

    args = make_args(
        urls_file=str(urls), html_dir=str(tmp_path / "html"), output=str(out),
        fetcher="stealthy", max_workers=8,
    )
    with caplog.at_level(logging.INFO, logger="web_reader"):
        scraper.run(args)
    assert any("clamping" in r.message for r in caplog.records)
