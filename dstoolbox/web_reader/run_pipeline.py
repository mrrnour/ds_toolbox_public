#!/usr/bin/env python3
"""Run the web_reader pipeline with optional bookmark pre-steps.

This script mirrors the previous shell pipeline while making argument parsing
and flow easier to read and maintain.

Sample runs  (activate .venv first: source .venv/bin/activate)
-----------
# Basic: scrape URLs from urls.txt and convert to Markdown
    python run_pipeline.py

# Specify a custom URL file
    python run_pipeline.py my_urls.txt

# Extract URLs from a Chrome Bookmarks JSON, then run the pipeline
    CHROME="$HOME/Library/Application Support/Google/Chrome/Default/Bookmarks"
    python run_pipeline.py -b "$CHROME" -f "statistical tests" -bl 100

# Filter a Netscape bookmarks export, then run the pipeline
    python run_pipeline.py -bi bookmarks_export.html -k "Tech" -k "AI"

# Full LinkedIn job pipeline: bookmark filter → URL extract → scrape → convert → rename
    CHROME="$HOME/Library/Application Support/Google/Chrome/Default/Bookmarks"
    python run_pipeline.py -bi bookmarks_export.html -k "Jobs" -b "$CHROME" -f "Jobs" -rj

# Dry-run rename step only (no other changes)
    python run_pipeline.py -rj -rd

# Reset checkpoint and rerun everything
    python run_pipeline.py -r
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import params

CHECKPOINT = Path(".pipeline_checkpoint")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for pipeline execution.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Read URLs, scrape/crawl, and convert outputs to Markdown."
    )
    parser.add_argument("-r", "--reset", action="store_true", help="Clear checkpoint and rerun all steps")

    parser.add_argument(
        "-bi",
        "--bookmarks-filter-input",
        default="",
        help="Input Netscape bookmarks HTML for tools/filter_bookmarks.py",
    )
    parser.add_argument(
        "-bo",
        "--bookmarks-filter-output",
        default="output/bookmarks_selected.html",
        help="Output filtered HTML path",
    )
    parser.add_argument(
        "-k",
        "--keep-folder",
        action="append",
        default=[],
        help="Folder name to keep during bookmark filtering (repeatable)",
    )
    parser.add_argument(
        "-f",
        "--folder-name",
        default="",
        help="Chrome bookmarks folder name to extract URLs from",
    )
    parser.add_argument(
        "-b",
        "--bookmarks-file",
        default=str(params.BOOKMARKS_FILE),
        help="Chrome Bookmarks JSON path for tools/bookmarks_to_urls.py",
    )
    parser.add_argument(
        "-bl",
        "--bookmarks-level",
        default="0",
        help="Subfolder recursion level for tools/bookmarks_to_urls.py",
    )

    parser.add_argument("urls_file", nargs="?", default="urls.txt", help="URL input file")

    parser.add_argument(
        "-rj",
        "--rename-to-jd",
        action="store_true",
        help="Optional last step: rename LinkedIn hash-named files to <JobTitle>_<Company> format",
    )
    parser.add_argument(
        "-rd",
        "--rename-dry-run",
        action="store_true",
        help="Pass --dry-run to rename_to_jd.py (preview only, no changes)",
    )
    parser.add_argument(
        "-ro",
        "--rename-output",
        default="",
        help="Custom output directory for rename_to_jd.py (default: pipeline output/)",
    )
    parser.add_argument(
        "--force-step",
        action="append",
        default=[],
        choices=["tools/filter_bookmarks.py", "tools/bookmarks_to_urls.py", "scraper.py", "convert.py", "tools/rename_to_jd.py"],
        help="Force rerun a specific step even if checkpoint says it is complete (repeatable)",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate argument combinations.

    Args:
        args: Parsed CLI arguments.
        parser: The parser used to report user-facing errors.

    Returns:
        None.
    """
    if args.bookmarks_filter_input and not args.keep_folder:
        parser.error("--bookmarks-filter-input requires at least one --keep-folder")

    if args.folder_name and not str(args.bookmarks_file).strip():
        parser.error("--folder-name requires --bookmarks-file or params.BOOKMARKS_FILE")


def load_checkpoint() -> set[str]:
    """Load completed step keys from checkpoint file.

    Returns:
        Set of completed step keys.
    """
    if not CHECKPOINT.exists():
        return set()
    return {line.strip() for line in CHECKPOINT.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_checkpoint(step_key: str) -> None:
    """Append a completed step key to checkpoint.

    Args:
        step_key: Unique step identifier.

    Returns:
        None.
    """
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(f"{step_key}\n")


def reset_pipeline() -> None:
    """Reset pipeline progress by removing checkpoint file.

    Returns:
        None.
    """
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
    print("Checkpoint cleared - pipeline will re-run from scratch")


def run_step(
    step_key: str,
    command: list[str],
    completed: set[str],
    check: bool = True,
    force: bool = False,
    required_paths: list[Path] | None = None,
) -> None:
    """Run one step unless checkpoint says it already completed.

    Args:
        step_key: Checkpoint key for the step.
        command: Executable command list.
        completed: Mutable set of completed step keys.
        check: If True, raise exception on non-zero exit. If False, allow non-zero exit.

    Returns:
        None.
    """
    if step_key in completed and not force:
        missing_outputs = [p for p in (required_paths or []) if not p.exists()]
        if not missing_outputs:
            print(f"Skipping {step_key} (checkpoint)")
            return
        missing_str = ", ".join(str(p) for p in missing_outputs)
        print(f"Checkpoint stale for {step_key}; missing output(s): {missing_str}. Re-running.")

    if force and step_key in completed:
        print(f"Running {step_key} (forced)")
    else:
        print(f"Running {step_key}")
    subprocess.run(command, check=check)
    append_checkpoint(step_key)
    completed.add(step_key)


def python_cmd(script: str, *script_args: str) -> list[str]:
    """Build a Python module command honoring the PYTHON env override.

    Translates a web_reader-relative script path (e.g. ``"scraper.py"``,
    ``"tools/filter_bookmarks.py"``) into the importable module path under
    ``dstoolbox.web_reader`` and invokes it via ``python -m`` so subprocesses
    resolve correctly regardless of the caller's working directory.

    Args:
        script: Script path relative to the ``dstoolbox/web_reader`` folder.
        script_args: Script arguments.

    Returns:
        Command list suitable for subprocess.run.
    """
    python_bin = os.environ.get("PYTHON", sys.executable)
    module = "dstoolbox.web_reader." + script.removesuffix(".py").replace("/", ".")
    return [python_bin, "-m", module, *script_args]


def main() -> int:
    """Execute the configured pipeline steps.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    forced_steps = set(args.force_step)

    if args.reset:
        reset_pipeline()

    completed = load_checkpoint()

    if args.bookmarks_filter_input:
        output_path = Path(args.bookmarks_filter_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = python_cmd(
            "tools/filter_bookmarks.py",
            "--input",
            args.bookmarks_filter_input,
            "--output",
            args.bookmarks_filter_output,
            "--keep",
            *args.keep_folder,
        )
        run_step(
            "tools/filter_bookmarks.py",
            cmd,
            completed,
            force="tools/filter_bookmarks.py" in forced_steps,
            required_paths=[output_path],
        )

    if args.folder_name:
        cmd = python_cmd(
            "tools/bookmarks_to_urls.py",
            "--folder-name",
            args.folder_name,
            "--bookmarks",
            args.bookmarks_file,
            "--level",
            str(args.bookmarks_level),
            "--output",
            args.urls_file,
        )
        run_step(
            "tools/bookmarks_to_urls.py",
            cmd,
            completed,
            force="tools/bookmarks_to_urls.py" in forced_steps,
            required_paths=[Path(args.urls_file)],
        )

    # Allow scraper to continue with partial failures (some URLs may be unreachable)
    run_step(
        "scraper.py",
        python_cmd("scraper.py", args.urls_file),
        completed,
        check=False,
        force="scraper.py" in forced_steps,
        required_paths=[Path(params.CRAWLING_SUMMARY_PATH)],
    )
    
    # Build convert command with optional folder name for organizing output
    convert_cmd = python_cmd("convert.py")
    if args.folder_name:
        convert_cmd.extend(["--folder", args.folder_name])
    run_step(
        "convert.py",
        convert_cmd,
        completed,
        force="convert.py" in forced_steps,
        required_paths=[Path(params.CONVERSION_SUMMARY_PATH)],
    )

    if args.rename_to_jd:
        rename_cmd = python_cmd("tools/rename_to_jd.py")
        if args.rename_dry_run:
            rename_cmd.append("--dry-run")
        if args.rename_output:
            rename_cmd.extend(["--output", args.rename_output])
        run_step(
            "tools/rename_to_jd.py",
            rename_cmd,
            completed,
            force="tools/rename_to_jd.py" in forced_steps,
        )

    print("Pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
