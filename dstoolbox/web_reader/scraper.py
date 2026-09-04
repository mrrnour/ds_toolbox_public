#!/usr/bin/env python3
"""URL scraper for the web_reader pipeline.

Reads a text file of URLs (one per line, # comments OK) and fetches each one
through one of two backends with the *same signature*, dispatched by `FETCHERS`:
  fetch_with_requests(url, cfg, session) -> FetchResult    (HTTP via requests)
  fetch_with_stealthy(url, cfg, session) -> FetchResult    (Camoufox via scrapling)

Output JSONL records use the shared `PipelineRecord` shape from utils.py so
convert.py can consume them directly. Each ok record carries `file_path`,
`depth` (always 0), `scraped_at` — the fields convert.py reads.

To scrape a Chrome bookmark folder: first run `bookmarks_to_urls.py
--folder-id N -o bookmarks.txt`, then `scraper.py bookmarks.txt`.

Defaults come from params.py.
"""

import argparse
import datetime as dt
import glob
import hashlib
import logging
import mimetypes
import os
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from . import params, utils
from .utils import PipelineRecord, record_to_dict, save_jsonl, setup_logger

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TEXT_EXTENSIONS = frozenset({".html", ".htm", ".xml", ".xhtml", ".txt", ".json", ".css", ".js"})


@dataclass
class FetchResult:
    """Uniform return type for both backends."""

    body: bytes | None
    content_type: str
    status_code: int | None
    error: str | None  # None on success; "empty_body" maps to skipped, others to failed


# A Fetcher is a backend-specific closure produced by a factory (see make_requests_fetcher,
# make_stealthy_fetcher). All backend config is bound at factory time.
Fetcher = Callable[[str], FetchResult]


@dataclass
class RequestsFetcherConfig:
    timeout_s: int = params.REQUEST_TIMEOUT
    max_retries: int = params.MAX_RETRIES


@dataclass
class StealthyFetcherConfig:
    timeout_ms: int = params.STEALTHY_TIMEOUT_MS
    headless: bool = params.HEADLESS
    network_idle: bool = params.NETWORK_IDLE


@dataclass
class ScrapeConfig:
    fetcher: str = params.DEFAULT_FETCHER
    output_path: Path = field(default_factory=lambda: Path(params.CRAWLING_SUMMARY_PATH))
    html_dir: Path = field(default_factory=lambda: Path(params.HTML_DIR))
    report_path: Path = field(default_factory=lambda: Path(params.REPORT_PATH))
    log_path: Path = field(default_factory=lambda: Path(params.LOG_FILE_PATH))
    source_name: str = ""
    max_workers: int = params.MAX_WORKERS
    delay: float = params.REQUEST_DELAY
    overwrite: bool = params.OVERWRITE


# === input ===
def items_from_file(path: Path) -> tuple[list[dict], str]:
    """Read URLs from a text file. Blank lines and lines starting with `#` are skipped.

    Special metadata comment: `# @folder_path: subfolder/nested` sets folder path for next URL.
    """
    if not path.exists():
        sys.exit(f"URLs file not found: {path}")
    utils.check_input_size(str(path))
    items: list[dict] = []
    current_folder_path = ""

    for raw in path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("#"):
            # Parse folder_path metadata
            if ln.startswith("# @folder_path:"):
                current_folder_path = ln.split("# @folder_path:", 1)[1].strip()
            continue

        item = {"url": ln}
        if current_folder_path:
            item["folder_path"] = current_folder_path
        items.append(item)

    return items, f"file {path}"


# === path helpers ===
def ext_for_response(url: str, content_type: str) -> str:
    """Pick a file extension from Content-Type, falling back to URL path."""
    ct = content_type.split(";")[0].strip().lower() if content_type else ""
    if ct == "text/html":
        return ".html"
    ext = mimetypes.guess_extension(ct) if ct else None
    if not ext:
        _, url_ext = os.path.splitext(urlparse(url).path)
        ext = url_ext or ".bin"
    return ext.lower()


def slug_for(url: str, ext: str = ".html") -> str:
    """Stable per-URL filename: <md5_10>_<sanitized_basename><ext>."""
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    base = os.path.basename(urlparse(url).path) or urlparse(url).netloc
    base_noext, _ = os.path.splitext(base)
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", base_noext)[:30].strip("._-") or "index"
    return f"{h}_{sanitized}{ext}"


def _retry_delay(headers: dict, attempt: int) -> float:
    ra = headers.get("Retry-After", "")
    try:
        return max(1.0, float(ra))
    except (ValueError, TypeError):
        return 2.0**attempt


# === fetcher factories — each builds a closure that holds backend-specific state ===
def make_requests_fetcher(cfg: RequestsFetcherConfig) -> Fetcher:
    """HTTP fetcher via requests. The closure owns its session. Retries 429 with Retry-After."""
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    def fetch(url: str) -> FetchResult:
        resp: requests.Response | None = None
        for attempt in range(cfg.max_retries + 1):
            try:
                resp = session.get(url, timeout=cfg.timeout_s)
            except requests.exceptions.RequestException as e:
                return FetchResult(None, "", None, f"request_error: {e}")
            if resp.status_code == 429 and attempt < cfg.max_retries:
                time.sleep(_retry_delay(resp.headers, attempt))
                continue
            break
        content_type = resp.headers.get("Content-Type", "") if resp is not None else ""
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            return FetchResult(None, content_type, resp.status_code, f"request_error: {e}")
        body = resp.content or b""
        if not body:
            return FetchResult(None, content_type, resp.status_code, "empty_body")
        return FetchResult(body, content_type, resp.status_code, None)

    return fetch


def make_stealthy_fetcher(cfg: StealthyFetcherConfig) -> Fetcher:
    """Stealth-browser fetcher via scrapling.StealthyFetcher (Camoufox). Single attempt per URL."""
    from scrapling.fetchers import StealthyFetcher  # heavy/optional dep — fail fast at factory time

    def fetch(url: str) -> FetchResult:
        try:
            page = StealthyFetcher.fetch(
                url,
                headless=cfg.headless,
                timeout=cfg.timeout_ms,
                network_idle=cfg.network_idle,
            )
        except Exception as e:
            return FetchResult(None, "", None, f"fetch_exception: {e}")
        if page is None:
            return FetchResult(None, "", None, "fetch_returned_none")
        status = getattr(page, "status", None)
        html = getattr(page, "html_content", "") or ""
        if status and status >= 400:
            return FetchResult(None, "text/html", status, f"http_{status}")
        if not html:
            return FetchResult(None, "text/html", status, "empty_body")
        return FetchResult(html.encode("utf-8"), "text/html", status, None)

    return fetch


FETCHERS: dict[str, Callable[..., Fetcher]] = {
    "requests": make_requests_fetcher,
    "stealthy": make_stealthy_fetcher,
}


# === per-URL pipeline ===
def process_one(item: dict, cfg: ScrapeConfig, fetcher: Fetcher) -> PipelineRecord:
    url = item["url"]
    name = item.get("name", "")
    folder_path = item.get("folder_path", "")  # From @folder_path metadata
    ts = dt.datetime.now().isoformat()

    # Build base data dict with optional folder_path
    base_data = {"name": name, "depth": 0, "scraped_at": ts}
    if folder_path:
        base_data["folder_path"] = folder_path

    if not cfg.overwrite:
        h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        existing = glob.glob(str(cfg.html_dir / f"{h}_*"))
        if existing:
            base_data["file_path"] = existing[0]
            return PipelineRecord.skipped(
                id=url,
                reason="already_exists",
                data=base_data,
            )

    if cfg.delay:
        time.sleep(cfg.delay)

    res = fetcher(url)

    if res.error == "empty_body":
        base_data.update({"status_code": res.status_code, "content_type": res.content_type})
        return PipelineRecord.skipped(
            id=url,
            reason=res.error,
            data=base_data,
        )
    if res.error:
        base_data.update({"status_code": res.status_code, "content_type": res.content_type})
        return PipelineRecord.failed(
            id=url,
            reason=res.error,
            data=base_data,
        )

    cfg.html_dir.mkdir(parents=True, exist_ok=True)
    ext = ext_for_response(url, res.content_type)
    file_path = cfg.html_dir / slug_for(url, ext=ext)
    if ext in TEXT_EXTENSIONS:
        file_path.write_text(res.body.decode("utf-8", errors="replace"), encoding="utf-8")
    else:
        file_path.write_bytes(res.body)

    base_data.update(
        {
            "file_path": str(file_path),
            "status_code": res.status_code,
            "content_length": len(res.body),
            "content_type": res.content_type,
        }
    )
    return PipelineRecord.ok(
        id=url,
        data=base_data,
    )


# === run ===
def run(args: argparse.Namespace, logger: logging.Logger | None = None) -> int:
    items, source_name = items_from_file(Path(args.urls_file))

    cfg = ScrapeConfig(
        fetcher=args.fetcher,
        output_path=Path(args.output),
        html_dir=Path(args.html_dir),
        report_path=Path(args.output).parent / "report.txt",
        log_path=Path(args.output).parent / "app.log",
        source_name=source_name,
        max_workers=args.max_workers,
        delay=args.delay,
        overwrite=args.overwrite,
    )

    if logger is None:
        logger = setup_logger(str(cfg.log_path), stream=False)

    if cfg.fetcher == "stealthy" and cfg.max_workers > 1:
        logger.info(f"clamping max_workers {cfg.max_workers} -> 1 for stealthy backend")
        cfg.max_workers = 1

    print(f"Source: {source_name}  URLs: {len(items)}  fetcher: {cfg.fetcher}")
    logger.info(f"source={source_name} count={len(items)} fetcher={cfg.fetcher}")

    if args.list_only:
        for it in items:
            print(it["url"])
        return 0

    if args.limit:
        items = items[: args.limit]
        print(f"(limited to first {args.limit})")
        logger.info(f"limited to first {args.limit}")

    cfg.html_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    started = dt.datetime.now().isoformat(timespec="seconds")
    if cfg.fetcher == "requests":
        fetcher = make_requests_fetcher(
            RequestsFetcherConfig(
                timeout_s=args.timeout_s,
                max_retries=args.max_retries,
            )
        )
    elif cfg.fetcher == "stealthy":
        fetcher = make_stealthy_fetcher(
            StealthyFetcherConfig(
                timeout_ms=args.timeout_ms,
                headless=args.headless,
                network_idle=args.network_idle,
            )
        )
    else:
        sys.exit(f"Unknown fetcher: {cfg.fetcher}")
    records: list[PipelineRecord] = []

    pbar = tqdm(total=len(items), desc="scrape", disable=not sys.stderr.isatty())
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as ex:
        futures = [ex.submit(process_one, it, cfg, fetcher) for it in items]
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            pbar.update(1)
            if rec.status == "failed":
                tqdm.write(f"failed: {rec.id} — {rec.reason}")
                logger.warning(f"failed: {rec.id} — {rec.reason}")
            elif rec.status == "skipped":
                logger.info(f"skipped: {rec.id} — {rec.reason}")
            else:
                logger.info(f"ok: {rec.id}")
    pbar.close()
    finished = dt.datetime.now().isoformat(timespec="seconds")

    save_jsonl([record_to_dict(r) for r in records], str(cfg.output_path), logger)
    utils.write_report(
        records,
        str(cfg.report_path),
        header_lines=[
            "=== scrape report ===",
            f"source:   {source_name}",
            f"fetcher:  {cfg.fetcher}",
            f"started:  {started}",
            f"finished: {finished}",
        ],
    )

    ok = sum(1 for r in records if r.status == "ok")
    skipped = sum(1 for r in records if r.status == "skipped")
    failed = sum(1 for r in records if r.status == "failed")
    print()
    print(f"ok={ok}  skipped={skipped}  failed={failed}  total={len(records)}")
    print(f"HTML:    {cfg.html_dir}/")
    print(f"Summary: {cfg.output_path}")
    print(f"Report:  {cfg.report_path}")
    print(f"Log:     {cfg.log_path}")
    return 0 if failed == 0 else 1


# === cli ===
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scrape URLs (one per line in a text file) to raw bytes."
    )
    p.add_argument(
        "urls_file",
        nargs="?",
        default=str(params.URLS_PATH),
        help=f"Text file with one URL per line; # comments OK (default: {params.URLS_PATH})",
    )

    f = p.add_argument_group("fetcher")
    f.add_argument(
        "--fetcher",
        choices=list(FETCHERS),
        default=params.DEFAULT_FETCHER,
        help=f"Backend (default: {params.DEFAULT_FETCHER})",
    )
    f.add_argument(
        "--timeout-s",
        type=int,
        default=params.REQUEST_TIMEOUT,
        help=f"requests timeout in seconds (default: {params.REQUEST_TIMEOUT})",
    )
    f.add_argument(
        "--timeout-ms",
        type=int,
        default=params.STEALTHY_TIMEOUT_MS,
        help=f"stealthy timeout in ms (default: {params.STEALTHY_TIMEOUT_MS})",
    )
    f.add_argument(
        "--max-workers",
        type=int,
        default=params.MAX_WORKERS,
        help=f"parallel workers (clamped to 1 for stealthy; default: {params.MAX_WORKERS})",
    )
    f.add_argument(
        "--max-retries",
        type=int,
        default=params.MAX_RETRIES,
        help=f"requests 429 retries (default: {params.MAX_RETRIES})",
    )
    f.add_argument(
        "--delay",
        type=float,
        default=params.REQUEST_DELAY,
        help="Sleep N seconds before each fetch (default: 0)",
    )

    st = p.add_argument_group("stealthy-only")
    st.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=params.HEADLESS,
        help="Run browser headless (default)",
    )
    st.add_argument(
        "--no-headless", dest="headless", action="store_false", help="Show browser window"
    )
    st.add_argument(
        "--network-idle",
        dest="network_idle",
        action="store_true",
        default=params.NETWORK_IDLE,
        help="Wait for network idle (default)",
    )
    st.add_argument(
        "--no-network-idle",
        dest="network_idle",
        action="store_false",
        help="Don't wait for network idle",
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "-o", "--output", default=str(params.CRAWLING_SUMMARY_PATH), help="JSONL summary path"
    )
    out.add_argument(
        "--html-dir", default=str(params.HTML_DIR), help="Directory for saved payloads"
    )

    pl = p.add_argument_group("pipeline")
    pl.add_argument("--limit", type=int, default=0, help="Stop after N URLs (0=no limit)")
    pl.add_argument(
        "--overwrite",
        action="store_true",
        default=params.OVERWRITE,
        help="Re-fetch even if a file with the same hash exists",
    )
    pl.add_argument("--list-only", action="store_true", help="Print URLs and exit (preview)")
    pl.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return p


def main() -> int:
    """Console entry point: parse args and run the scraper."""
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
