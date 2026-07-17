"""Tests for bookmarks_to_urls.py."""

import json

import pytest

import dstoolbox.web_reader.tools.bookmarks_to_urls as b2u


# === pure tree helpers ===

def test_collect_urls_recursion_level():
    folder = {
        "type": "folder",
        "id": "1",
        "children": [
            {"type": "url", "url": "https://a.example/1", "name": "1"},
            {
                "type": "folder", "id": "2", "name": "sub",
                "children": [
                    {"type": "url", "url": "https://a.example/sub1", "name": "sub1"},
                    {
                        "type": "folder", "id": "3", "name": "subsub",
                        "children": [
                            {"type": "url", "url": "https://a.example/ss1", "name": "ss1"},
                        ],
                    },
                ],
            },
        ],
    }
    assert len(b2u.collect_urls(folder, max_level=0)) == 1
    assert len(b2u.collect_urls(folder, max_level=1)) == 2
    assert len(b2u.collect_urls(folder, max_level=2)) == 3
    assert len(b2u.collect_urls(folder, max_level=-1)) == 3


def test_load_folder_by_name_finds_case_insensitive(tmp_path):
    bookmarks = {
        "roots": {
            "bookmark_bar": {
                "type": "folder",
                "id": "0",
                "name": "root",
                "children": [
                    {
                        "type": "folder",
                        "id": "42",
                        "name": "Statistical Tests",
                        "children": [],
                    }
                ],
            }
        }
    }
    bf = tmp_path / "Bookmarks"
    bf.write_text(json.dumps(bookmarks))
    folder = b2u.load_folder_by_name(bf, "statistical tests")
    assert folder["id"] == "42"


def test_load_folder_by_name_duplicate_exits(tmp_path):
    bookmarks = {
        "roots": {
            "bookmark_bar": {
                "type": "folder",
                "id": "0",
                "name": "root",
                "children": [
                    {"type": "folder", "id": "42", "name": "same", "children": []},
                    {"type": "folder", "id": "99", "name": "same", "children": []},
                ],
            }
        }
    }
    bf = tmp_path / "Bookmarks"
    bf.write_text(json.dumps(bookmarks))
    with pytest.raises(SystemExit):
        b2u.load_folder_by_name(bf, "same")


# === render ===

def test_render_writes_name_comment_above_url():
    items = [
        {"url": "https://a/1", "name": "First post"},
        {"url": "https://a/2", "name": "Second"},
    ]
    out = b2u.render(items, "folder X name='X'")
    lines = out.splitlines()
    assert lines[0].startswith("# folder X")
    assert "# First post" in lines
    assert "https://a/1" in lines
    assert "# Second" in lines
    assert "https://a/2" in lines


def test_render_handles_url_without_name():
    items = [{"url": "https://a/1", "name": ""}]
    out = b2u.render(items, "h")
    assert "https://a/1" in out


def test_render_strips_newlines_in_names():
    items = [{"url": "https://a/1", "name": "Title\nwith\rnewlines"}]
    out = b2u.render(items, "h")
    assert "# Title with newlines" in out
    assert "\n# Title\nwith" not in out


def test_render_empty_folder():
    out = b2u.render([], "h")
    assert "# h" in out
    assert "https://" not in out


# === scraper integration: the rendered file must round-trip through items_from_file ===

def test_rendered_file_feeds_into_scraper(tmp_path):
    from dstoolbox.web_reader import scraper

    items = [
        {"url": "https://a/1", "name": "First"},
        {"url": "https://a/2", "name": "Second"},
    ]
    text = b2u.render(items, "test header")
    f = tmp_path / "bookmarks.txt"
    f.write_text(text)

    parsed, _ = scraper.items_from_file(f)
    urls = [it["url"] for it in parsed]
    assert urls == ["https://a/1", "https://a/2"]


# === CLI ===

def test_cli_requires_folder_selector_or_list():
    parser = b2u.build_parser()
    args = parser.parse_args([])
    with pytest.raises(SystemExit):
        b2u.validate_args(args, parser)


def test_cli_allows_folder_name():
    parser = b2u.build_parser()
    args = parser.parse_args(["--folder-name", "Statistical Tests"])
    b2u.validate_args(args, parser)
