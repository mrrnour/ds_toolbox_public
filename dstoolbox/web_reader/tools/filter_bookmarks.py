#!/usr/bin/env python3
"""Filter a Netscape bookmarks HTML file by keeping selected folders.

Example:
  python3 filter_bookmarks.py \
    --input bookmarks_6_16_26.html \
    --output bookmarks_statistical_tests_only.html \
    --keep "statistical tests"

  python3 filter_bookmarks.py \
    -i bookmarks_6_16_26.html \
    -o bookmarks_selected.html \
    --keep "statistical tests" "A/B test"
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from collections.abc import Iterable
from pathlib import Path

H3_RE = re.compile(r"<H3\b[^>]*>(.*?)</H3>", re.IGNORECASE)


def normalize_name(name: str) -> str:
    return " ".join(html.unescape(name).strip().lower().split())


def extract_h3_text(line: str) -> str | None:
    match = H3_RE.search(line)
    if not match:
        return None
    return html.unescape(match.group(1)).strip()


def find_folder_block(lines: list[str], folder_start_idx: int) -> tuple[int, int] | None:
    """Return (start_idx, end_idx) for one folder block including nested content.

    Expects the folder to be represented as:
      <DT><H3 ...>Folder</H3>
      <DL><p>
        ...
      </DL><p>
    """
    open_idx = None
    for idx in range(folder_start_idx + 1, len(lines)):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if "<DL><p>" in stripped:
            open_idx = idx
        break

    if open_idx is None:
        return None

    depth = 0
    end_idx = None
    for idx in range(open_idx, len(lines)):
        line = lines[idx]
        depth += line.count("<DL><p>")
        depth -= line.count("</DL><p>")
        if depth == 0:
            end_idx = idx
            break

    if end_idx is None:
        return None

    return folder_start_idx, end_idx


def collect_available_folders(lines: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for line in lines:
        if "<H3" not in line.upper():
            continue
        name = extract_h3_text(line)
        if not name:
            continue
        key = normalize_name(name)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(name)
    return ordered


def build_output(blocks: list[list[str]]) -> str:
    header = [
        "<!DOCTYPE NETSCAPE-Bookmark-file-1>",
        "<!-- This is an automatically generated file.",
        "     It will be read and overwritten.",
        "     DO NOT EDIT! -->",
        '<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">',
        "<TITLE>Bookmarks</TITLE>",
        "<H1>Bookmarks</H1>",
        "<DL><p>",
        '    <DT><H3 ADD_DATE="0" LAST_MODIFIED="0" PERSONAL_TOOLBAR_FOLDER="true">Bookmarks Bar</H3>',
        "    <DL><p>",
    ]

    footer = [
        "    </DL><p>",
        "</DL><p>",
    ]

    out_lines = header[:]
    for block in blocks:
        out_lines.extend(["        " + line for line in block])
    out_lines.extend(footer)
    return "\n".join(out_lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only selected bookmark folders from a Netscape bookmarks HTML file."
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input bookmarks HTML file")
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output filtered bookmarks HTML file"
    )
    parser.add_argument(
        "--keep",
        nargs="+",
        required=False,
        help="Folder names to keep (case-insensitive). You can pass one or many.",
    )
    parser.add_argument(
        "--list-folders",
        action="store_true",
        help="List all folder names in the input file and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    lines = args.input.read_text(encoding="utf-8", errors="replace").splitlines()

    if args.list_folders:
        for name in collect_available_folders(lines):
            print(name)
        return 0

    if not args.keep:
        print(
            "You must pass at least one folder name with --keep, or use --list-folders.",
            file=sys.stderr,
        )
        return 1

    keep_set = {normalize_name(name) for name in args.keep}

    selected_blocks: list[list[str]] = []
    seen_ranges: set[tuple[int, int]] = set()

    for idx, line in enumerate(lines):
        if "<H3" not in line.upper():
            continue
        folder_name = extract_h3_text(line)
        if not folder_name:
            continue
        if normalize_name(folder_name) not in keep_set:
            continue

        rng = find_folder_block(lines, idx)
        if rng is None:
            continue
        if rng in seen_ranges:
            continue
        seen_ranges.add(rng)
        start, end = rng
        selected_blocks.append(lines[start : end + 1])

    if not selected_blocks:
        available = collect_available_folders(lines)
        print("No matching folder names were found for --keep.", file=sys.stderr)
        if available:
            print("\nAvailable folders:", file=sys.stderr)
            for name in available:
                print(f"  - {name}", file=sys.stderr)
        return 2

    output_text = build_output(selected_blocks)
    args.output.write_text(output_text, encoding="utf-8")

    print(f"Created {args.output}")
    print(f"Folders kept: {len(selected_blocks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
