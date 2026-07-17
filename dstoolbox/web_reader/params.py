"""Defaults for the web_reader pipeline (scraper.py + convert.py + harvest.py + rename_to_jd.py)."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# === paths ===
URLS_PATH                = HERE / "urls.txt"
OUTPUT_DIR               = HERE / "output"
HTML_DIR                 = OUTPUT_DIR / "html"
MARKDOWN_DIR             = OUTPUT_DIR / "markdown"
FILES_DIR                = OUTPUT_DIR / "files"
CRAWLING_SUMMARY_PATH    = OUTPUT_DIR / "crawling_summary.jsonl"
CONVERSION_SUMMARY_PATH  = OUTPUT_DIR / "conversion_summary.jsonl"
HARVEST_SUMMARY_PATH     = OUTPUT_DIR / "harvest_summary.jsonl"
REPORT_PATH              = OUTPUT_DIR / "report.txt"
LOG_FILE_PATH            = OUTPUT_DIR / "app.log"
BOOKMARKS_FILE           = os.environ.get(
	"CHROME",
	str(Path.home() / "Library/Application Support/Google/Chrome/Default/Bookmarks"),
)

# === fetcher ===
DEFAULT_FETCHER          = "requests"   # "requests" | "stealthy"

# === network — requests backend ===
REQUEST_TIMEOUT          = 30           # seconds
MAX_RETRIES              = 3

# === network — stealthy backend ===
STEALTHY_TIMEOUT_MS      = 60_000
HEADLESS                 = True
NETWORK_IDLE             = True

# === pipeline ===
MAX_WORKERS              = 4            # auto-clamped to 1 for stealthy
REQUEST_DELAY            = 0.0
BOOKMARK_LEVEL           = 0            # 0=selected only, -1=unlimited recursion
MAX_INPUT_MB             = 100          # urls-file size guard
OVERWRITE                = False

# === convert.py ===
MIN_MARKDOWN_LENGTH      = 200
