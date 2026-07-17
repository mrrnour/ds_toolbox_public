#!/usr/bin/env python3
"""Extract a Chrome bookmark folder to a text file (one URL per line).

Each URL is preceded by a `# <name>` comment so the file is human-scannable.
Comments and blank lines are stripped by scraper.items_from_file, so the same
file feeds straight into scraper.py.

Usage:
    python bookmarks_to_urls.py --folder-name "Statistical Tests" -o bookmarks.txt
    python bookmarks_to_urls.py --folder-name "Statistical Tests" --level 1 | tee bookmarks.txt
    python bookmarks_to_urls.py --folder-name "Statistical Tests" --bookmarks /path/to/Bookmarks
    python bookmarks_to_urls.py --list-folders
"""

import argparse
import json
import re
import sys
from pathlib import Path

from dstoolbox.web_reader import params


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def normalize_folder_component(name: str) -> str:
    """Normalize folder names to filesystem-safe kebab-case."""
    value = " ".join(str(name).strip().split()).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "untitled"


def collect_folders(node: dict, out: list[dict]) -> None:
    if not isinstance(node, dict):
        return
    if node.get("type") == "folder":
        out.append({"name": str(node.get("name", ""))})
    for child in node.get("children", []) or []:
        collect_folders(child, out)


def find_folders_by_name(node: dict, target_name: str, out: list[dict]) -> None:
    if not isinstance(node, dict):
        return
    if node.get("type") == "folder" and normalize_name(str(node.get("name", ""))) == target_name:
        out.append(node)
    for child in node.get("children", []) or []:
        find_folders_by_name(child, target_name, out)


def collect_urls(folder: dict, max_level: int, _current: int = 0, _path: str = "") -> list[dict]:
    """0=selected folder only, 1=+immediate subfolders, N=descend N, -1=unlimited.
    
    Returns list of dicts with 'url', 'name', and 'folder_path' keys.
    folder_path is the hierarchy path like 'subfolder/nested'.
    """
    can_recurse = max_level < 0 or _current < max_level
    out: list[dict] = []
    for child in folder.get("children", []) or []:
        ctype = child.get("type")
        child_name = str(child.get("name", "")).strip()
        if ctype == "url":
            out.append({
                "url": child.get("url", ""),
                "name": child.get("name", ""),
                "folder_path": _path
            })
        elif ctype == "folder" and can_recurse:
            safe_child_name = normalize_folder_component(child_name)
            child_path = f"{_path}/{safe_child_name}" if _path else safe_child_name
            out.extend(collect_urls(child, max_level, _current + 1, child_path))
    return out


def load_bookmarks_data(bookmarks_path: Path) -> dict:
    if not bookmarks_path.exists():
        sys.exit(f"Bookmarks file not found: {bookmarks_path}")
    return json.loads(bookmarks_path.read_text(encoding="utf-8"))


def list_all_folders(data: dict) -> list[dict]:
    folders: list[dict] = []
    for root in data.get("roots", {}).values():
        collect_folders(root, folders)
    return folders


def load_folder_by_name(bookmarks_path: Path, folder_name: str) -> dict:
    data = load_bookmarks_data(bookmarks_path)
    target_name = normalize_name(folder_name)
    matches: list[dict] = []
    for root in data.get("roots", {}).values():
        find_folders_by_name(root, target_name, matches)

    if not matches:
        sys.exit(f"Folder name={folder_name!r} not found in {bookmarks_path}")

    if len(matches) > 1:
        details = ", ".join(f"name={m.get('name', '')!r}" for m in matches)
        sys.exit(
            "Multiple folders match the same name. "
            f"Please use a unique folder name. Matches: {details}"
        )

    return matches[0]


def render(items: list[dict], header: str) -> str:
    lines = [f"# {header}"]
    if not items:
        return "\n".join(lines) + "\n"
    lines.append("")
    for it in items:
        name = (it.get("name") or "").replace("\n", " ").replace("\r", " ").strip()
        if name:
            lines.append(f"# {name}")
        # Add folder_path as metadata comment for scraper to parse
        folder_path = it.get("folder_path", "")
        if folder_path:
            lines.append(f"# @folder_path: {folder_path}")
        lines.append(it["url"])
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    bookmarks_path = Path(args.bookmarks)

    if args.list_folders:
        data = load_bookmarks_data(bookmarks_path)
        for folder in list_all_folders(data):
            print(folder["name"])
        return 0

    folder = load_folder_by_name(bookmarks_path, args.folder_name)
    selected = f"name={args.folder_name!r}"

    items = collect_urls(folder, max_level=args.level)
    header = (
        f"folder {folder.get('name', '')!r}  {selected}  "
        f"level={args.level}  count={len(items)}"
    )
    output = render(items, header)
    if args.output and args.output != "-":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"wrote {len(items)} URLs to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract a Chrome bookmark folder to a urls.txt-compatible file."
    )
    p.add_argument("--folder-name",
                   help='Bookmark folder name (case-insensitive), e.g. "Statistical Tests"')
    p.add_argument("--list-folders", action="store_true",
                   help="List folder names from the bookmarks file, then exit")
    p.add_argument("--bookmarks", default=str(params.BOOKMARKS_FILE),
                   help="Path to Chrome Bookmarks JSON")
    p.add_argument("--level", type=int, default=params.BOOKMARK_LEVEL, metavar="N",
                   help="Subfolder recursion: 0=selected only (default), 1=+immediate, N=descend N, -1=unlimited")
    p.add_argument("-o", "--output", default="-",
                   help="Output file ('-' = stdout, default)")
    return p


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.list_folders:
        return
    if not args.folder_name:
        parser.error("--folder-name is required unless --list-folders is used")


if __name__ == "__main__":
    parser = build_parser()
    parsed = parser.parse_args()
    validate_args(parsed, parser)
    sys.exit(run(parsed))
