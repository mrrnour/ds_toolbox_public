# `dstoolbox.web_reader`

CLI pipeline to extract URLs (optionally from Chrome bookmarks), scrape
content, and convert to Markdown.

> Previously shipped as a standalone folder with its own `pyproject.toml` /
> `.venv`. It's now a first-class subpackage of `dstoolbox` — install it
> with the `[webreader]` extra and invoke everything either as a module
> (`python -m dstoolbox.web_reader.<name>`) or via the registered console
> scripts (`webreader-scrape`, `webreader-harvest`, `webreader-convert`,
> `webreader-pipeline`).

## Setup

```bash
pip install -e ".[webreader]"                 # from the dstoolbox repo root
# Optional stealth backend (scrapling, curl_cffi, patchright, msgspec, camoufox):
pip install -e ".[webreader-stealth]"
```

## Quick Start

Run with a URL list (module form):

```bash
python -m dstoolbox.web_reader.run_pipeline urls.txt
# or, equivalently, via the console script:
webreader-pipeline urls.txt
```

Run from a Chrome bookmarks folder (uses `params.BOOKMARKS_FILE` by default):

```bash
webreader-pipeline -f "statistical tests" -bl 100
```

Force only the convert stage without clearing all checkpoints:

```bash
webreader-pipeline -f "statistical tests" -bl 100 --force-step convert.py
```

Re-run everything from scratch:

```bash
webreader-pipeline -r -f "statistical tests" -bl 100
```

## Common Workflow Examples

Use these copy-paste commands for the statistical-tests bookmark flow:

Run full pipeline from bookmarks folder (includes deep subfolders):

```bash
webreader-pipeline -f "statistical tests" -bl 100
```

Force only conversion (keep URL extraction + scraping checkpointed):

```bash
webreader-pipeline -f "statistical tests" -bl 100 --force-step convert.py
```

Force scraping and conversion together:

```bash
webreader-pipeline -f "statistical tests" -bl 100 \
    --force-step scraper.py --force-step convert.py
```

Use an explicit bookmarks file path (portable across machines):

```bash
CHROME="$HOME/Library/Application Support/Google/Chrome/Default/Bookmarks"
webreader-pipeline -b "$CHROME" -f "statistical tests" -bl 100
```

Reset everything and rerun all stages:

```bash
webreader-pipeline -r -f "statistical tests" -bl 100
```

## Main Commands

| Console script | Module invocation | Purpose |
| --- | --- | --- |
| `webreader-pipeline` | `python -m dstoolbox.web_reader.run_pipeline` | Orchestrate all stages with checkpointing. |
| `webreader-scrape` | `python -m dstoolbox.web_reader.scraper` | Fetch HTML / PDF for a URL list. |
| `webreader-harvest` | `python -m dstoolbox.web_reader.harvest` | Auth-aware harvester (basic / NTLM). |
| `webreader-convert` | `python -m dstoolbox.web_reader.convert` | Convert crawled HTML / PDF → Markdown. |

### `run_pipeline` (recommended)

Runs, in order:
1. `tools.bookmarks_to_urls` (when `-f/--folder-name` is provided).
2. `scraper`.
3. `convert`.
4. `tools.rename_to_jd` (optional, when `-rj` is provided).

Sub-steps are executed as `python -m dstoolbox.web_reader.<mod>`
subprocesses, so they work regardless of the caller's CWD.

Key flags:
- `-r, --reset`: clear `.pipeline_checkpoint` and rerun all stages
- `-f, --folder-name`: Chrome bookmarks folder name to extract URLs from
- `-b, --bookmarks-file`: bookmarks JSON path (defaults to `params.BOOKMARKS_FILE`)
- `-bl, --bookmarks-level`: recursion depth for subfolders (`-1` unlimited)
- `--force-step`: force rerun a checkpointed stage (repeatable)

Allowed `--force-step` values (still spelled as script paths for
backward compatibility with earlier docs):
- `tools/filter_bookmarks.py`
- `tools/bookmarks_to_urls.py`
- `scraper.py`
- `convert.py`
- `tools/rename_to_jd.py`

### `scraper`

Reads URL list and fetches content. Failures are recorded per URL and the
run continues.

Key flags:
- `--fetcher {requests,stealthy}`
- `--max-workers`
- `--max-retries`
- `--delay`
- `--output`
- `--html-dir`
- `--overwrite`

### `convert`

Converts crawled files to Markdown and writes a conversion summary.

Supported formats:
- `.html`, `.htm` via `markdownify`
- `.pdf` via `pypdf`
- optional docling mode for additional formats (`--use-docling`)

Key flags:
- `--output-dir`
- `--folder`
- `--min-length`
- `--no-overwrite`
- `--use-docling`
- `--image`

## Naming and Folder Rules

### Folder structure

- Bookmark folder paths are normalized to safe kebab-case components.
- Nested bookmark folders are preserved in output directories.
- Example: `A/B test` becomes `a-b-test`.

### File naming

- Markdown uses readable names: `<category>-<title-or-url-slug>.md`.
- HTML mirrors use exactly the same stem/path: `<same-stem>.html`.
- If duplicates occur in the same folder, numeric prefixes are added:
  - `name.md`
  - `2-name.md`
  - `3-name.md`

Legacy hash-style HTML names are cleaned up during conversion.

## Checkpoint Behavior

Pipeline progress is tracked in `.pipeline_checkpoint`.

- If a step key exists in checkpoint, it is skipped.
- If checkpoint says complete but required output is missing, step reruns
  automatically (stale-checkpoint protection).
- Use `--force-step` for targeted reruns.

## Config (`params.py`)

Important defaults:
- `BOOKMARKS_FILE`: defaults to `$CHROME` env var, otherwise macOS Chrome default path
- `OUTPUT_DIR`, `HTML_DIR`, `MARKDOWN_DIR`
- `CRAWLING_SUMMARY_PATH`, `CONVERSION_SUMMARY_PATH`
- `REQUEST_TIMEOUT`, `MAX_RETRIES`, `MAX_WORKERS`
- `MIN_MARKDOWN_LENGTH`

## Output Files

- `output/html/...`: mirrored HTML files with same naming convention as markdown
- `output/crawling_summary.jsonl`: one scrape `PipelineRecord` per URL
- `output/markdown/...`: converted markdown files
- `output/conversion_summary.jsonl`: one convert `PipelineRecord` per URL
- `output/report.txt`: latest report
- `output/app.log`: runtime logs

## Troubleshooting

- `Skipping ... (checkpoint)`: expected when step already completed; use
  `-r` or `--force-step <step>` to rerun.
- `Error: input not found: output/crawling_summary.jsonl`: stale/missing
  artifacts; rerun scraper or run with `--force-step scraper.py`.
- `ModuleNotFoundError`: install the `[webreader]` extra
  (`pip install -e ".[webreader]"`) and activate the correct virtualenv.

## Testing

From the dstoolbox repo root:

```bash
pytest dstoolbox/web_reader/tests -q
```
