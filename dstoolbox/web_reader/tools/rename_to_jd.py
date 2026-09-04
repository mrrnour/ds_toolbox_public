#!/usr/bin/env python3
"""Post-pipeline patch: rename `<hash>_www.linkedin.{md,html}` files to
`<JobTitle>_<Company>.{md,html}`, mirroring the JD/PDF naming convention
used downstream.

Standalone — does NOT modify crawl.py / convert.py. Intended to run after
`bash run_pipeline.sh ...` completes.

Behavior:
  - Walks output/markdown/*.md
  - Parses each file's H1 (job title) and topcard `[Company](url)` (company)
  - Computes `<Job_Title>_<Company>.md` (snake_case, sanitized)
  - Renames the markdown
  - Renames the matching .html file (same hash prefix → output/html/<hash>_*.html)
  - Updates output/crawling_summary.jsonl and output/conversion_summary.jsonl
    so file_path entries point to the new names
  - Skips files that are already in JD-name format (idempotent)
  - Skips files that don't have a parseable H1 + company (e.g. login walls,
    stale unrelated markdowns) — they're left untouched and reported

Usage:
  python rename_to_jd.py                # rename in place
  python rename_to_jd.py --dry-run      # preview only, no changes
  python rename_to_jd.py --output DIR   # operate on a non-default output dir
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HASH_PREFIX = re.compile(r"^[0-9a-f]{8,}_www\.linkedin")  # only rename hash-named files
H1 = re.compile(r"^# (.+?)\s*$", re.MULTILINE)
# topcard line:  #### [Company](url) Location ...
TOPCARD_COMPANY = re.compile(r"^####\s*\[([^\]]+)\]\(", re.MULTILINE)


def sanitize(s: str) -> str:
    """Match the snake_case style used in new_jobs/JD/ filenames."""
    s = s.strip()
    # Drop common LinkedIn suffix noise
    s = re.sub(r"\s*\|\s*LinkedIn\s*$", "", s, flags=re.IGNORECASE)
    # Replace separators with spaces
    s = re.sub(r"[/&\-,():\.]+", " ", s)
    # Replace anything not [A-Za-z0-9_ ] with space
    s = re.sub(r"[^A-Za-z0-9_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def parse_md(path: Path) -> tuple[str | None, str | None]:
    """Return (job_title, company) or (None, None) if unparseable."""
    text = path.read_text(encoding="utf-8", errors="replace")
    m_title = H1.search(text)
    m_company = TOPCARD_COMPANY.search(text)
    if not m_title or not m_company:
        return None, None
    title = m_title.group(1).strip()
    company = m_company.group(1).strip()
    if not title or not company:
        return None, None
    return title, company


def plan(md_dir: Path, html_dir: Path) -> list[dict]:
    """Build a rename plan. Each entry: {old_md, new_md, old_html, new_html, reason?}."""
    plan: list[dict] = []
    seen_targets: set[str] = set()
    for md in sorted(md_dir.glob("*.md")):
        if not HASH_PREFIX.match(md.stem):
            # Already renamed or unrelated file — skip silently
            continue
        title, company = parse_md(md)
        if not title or not company:
            plan.append(dict(old_md=md, reason="no H1 or topcard match"))
            continue

        base = f"{sanitize(title)}_{sanitize(company)}"
        if not base or base == "_":
            plan.append(dict(old_md=md, reason="empty sanitized name"))
            continue

        # De-duplicate: if two postings produce the same name, suffix _2, _3...
        candidate = base
        n = 2
        while (
            candidate in seen_targets
            or (md_dir / f"{candidate}.md").exists()
            and (md_dir / f"{candidate}.md") != md
        ):
            candidate = f"{base}_{n}"
            n += 1
        seen_targets.add(candidate)

        new_md = md_dir / f"{candidate}.md"

        # Find matching HTML (same hash prefix). Could be hash_www.linkedin.html.
        hash_prefix = md.stem  # e.g. 02074ba598_www.linkedin
        old_html_candidates = list(html_dir.glob(f"{hash_prefix}.html"))
        old_html = old_html_candidates[0] if old_html_candidates else None
        new_html = html_dir / f"{candidate}.html" if old_html else None

        plan.append(
            dict(
                old_md=md,
                new_md=new_md,
                old_html=old_html,
                new_html=new_html,
                title=title,
                company=company,
            )
        )
    return plan


def update_jsonl(path: Path, mapping: dict[str, str], dry_run: bool) -> int:
    """Rewrite a JSONL file, replacing any old paths in `mapping` with new paths.
    Updates `data.file_path` (crawling summary) and `data.md_path` (conversion
    summary) — whichever is present. Returns count of records changed."""
    if not path.exists():
        return 0
    changed = 0
    new_lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            new_lines.append(raw)
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            new_lines.append(raw)
            continue
        data = rec.get("data") or {}
        for field in ("file_path", "md_path"):
            fp = data.get(field)
            if not isinstance(fp, str):
                continue
            base = Path(fp).name
            if base in mapping:
                data[field] = str(Path(fp).with_name(mapping[base]))
                changed += 1
        new_lines.append(json.dumps(rec, ensure_ascii=False))
    if changed and not dry_run:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", default="./output", help="output directory (default: ./output)")
    ap.add_argument("--dry-run", action="store_true", help="preview only — don't rename")
    args = ap.parse_args()

    out = Path(args.output).resolve()
    md_dir = out / "markdown"
    html_dir = out / "html"
    if not md_dir.is_dir():
        print(f"ERROR: markdown dir not found: {md_dir}", file=sys.stderr)
        return 2

    items = plan(md_dir, html_dir)
    renames = [it for it in items if "new_md" in it]
    skips = [it for it in items if "reason" in it]

    if skips:
        print(f"⚠ {len(skips)} file(s) skipped (could not parse title/company):")
        for it in skips:
            print(f"  - {it['old_md'].name}: {it['reason']}")

    if not renames:
        print("Nothing to rename.")
        return 0

    print(f"\n{'PREVIEW' if args.dry_run else 'RENAMING'} {len(renames)} file(s):")
    for it in renames:
        print(f"  md:   {it['old_md'].name}")
        print(f"     →  {it['new_md'].name}")
        if it.get("old_html"):
            print(f"  html: {it['old_html'].name}")
            print(f"     →  {it['new_html'].name}")

    if args.dry_run:
        print("\n(dry-run; no changes made)")
        return 0

    # Apply renames
    md_map: dict[str, str] = {}
    html_map: dict[str, str] = {}
    for it in renames:
        old_md: Path = it["old_md"]
        new_md: Path = it["new_md"]
        if old_md != new_md:
            old_md.rename(new_md)
            md_map[old_md.name] = new_md.name
        if it.get("old_html") and it.get("new_html"):
            old_html: Path = it["old_html"]
            new_html: Path = it["new_html"]
            if old_html != new_html:
                old_html.rename(new_html)
                html_map[old_html.name] = new_html.name

    # Update JSONL summaries
    crawl_changed = update_jsonl(out / "crawling_summary.jsonl", html_map, dry_run=False)
    conv_changed = update_jsonl(
        out / "conversion_summary.jsonl", {**html_map, **md_map}, dry_run=False
    )

    print(f"\n✓ Renamed {len(md_map)} markdown, {len(html_map)} html")
    print(f"✓ Updated {crawl_changed} record(s) in crawling_summary.jsonl")
    print(f"✓ Updated {conv_changed} record(s) in conversion_summary.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
