#!/usr/bin/env python3
"""Harvest matching files (e.g. PDFs) from authenticated intranet listing pages.

Each input URL is treated as an *index page*: ``harvest.py`` parses its
``<a href>`` links and downloads only those whose filenames end in one of
``--extensions`` into a per-URL subdirectory under ``--output-dir``. A
sidecar ``<file>.meta.json`` lands next to each saved file with provenance.

Differs from :mod:`scraper`:
    * ``scraper.py``  fetches each URL itself.
    * ``harvest.py`` treats each URL as a directory listing and pulls files off it.

Skip-and-continue on errors. Retries 429s with ``Retry-After`` + exponential
backoff. Streams large files with byte-level progress.

Auth (intranet use case): set ``--auth basic`` or ``--auth ntlm`` and put
``WEB_READER_USER`` / ``WEB_READER_PASSWORD`` in ``.env``.
"""

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from . import params, utils
from .utils import PipelineRecord, record_to_dict


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class HarvestConfig:
    extensions:    List[str]
    auth_type:     str
    output_dir:    str
    max_workers:   int
    timeout:       int
    chunk_size:    int
    verify_ssl:    bool
    overwrite:     bool  = False
    request_delay: float = 0.0
    max_retries:   int   = 3


# ---------------------------------------------------------------------- helpers
def load_pages(path: str, logger: logging.Logger) -> list[str]:
    """Read listing-page URLs (one per line; '#' comments OK)."""
    if not os.path.exists(path):
        sys.exit(f"Error: input not found: {path}")
    utils.check_input_size(path)
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            out.append(ln)
    logger.info(f"Loaded {len(out)} listing URL(s) from {path}")
    return out


def sanitize_path(name: str) -> str:
    """Replace anything not [A-Za-z0-9._-] with '_', cap at 60 chars."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return s[:60] or "idx"


def short_slug(text: str, ext: str = "") -> str:
    """``<md5[:10]>_<sanitized[:30]><ext>`` — used for the path-length fallback."""
    h    = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    base = sanitize_path(text)[:30] or "idx"
    return f"{h}_{base}{ext}"


def _retry_delay(headers: Dict[str, str], attempt: int) -> float:
    ra = headers.get("Retry-After", "")
    try:
        return max(1.0, float(ra))
    except (ValueError, TypeError):
        return 2.0 ** attempt


def setup_session(auth_type: str, verify_ssl: bool) -> requests.Session:
    """Build a ``requests.Session`` with browser headers + optional Basic/NTLM auth.

    Credentials come from env vars: ``WEB_READER_USER`` and
    ``WEB_READER_PASSWORD`` (load_dotenv() runs at module import).
    """
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    session.verify = verify_ssl
    if not verify_ssl:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if auth_type == "none":
        return session

    user = os.environ.get("WEB_READER_USER", "")
    pwd  = os.environ.get("WEB_READER_PASSWORD", "")
    if not user or not pwd:
        sys.exit("Error: --auth requires WEB_READER_USER + WEB_READER_PASSWORD env vars (use .env).")

    if auth_type == "basic":
        session.auth = (user, pwd)
    elif auth_type == "ntlm":
        try:
            from requests_ntlm import HttpNtlmAuth
        except ImportError:
            sys.exit("Error: --auth ntlm requires `pip install requests-ntlm`.")
        session.auth = HttpNtlmAuth(user, pwd)
    else:
        sys.exit(f"Error: unknown auth_type {auth_type!r}; use none|basic|ntlm.")
    return session


def list_files_on_page(
    session: requests.Session, page_url: str, extensions: List[str], timeout: int,
) -> List[Tuple[str, str]]:
    """Fetch the listing page, return ``[(absolute_file_url, filename), ...]``."""
    resp = session.get(page_url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in extensions)
    out: List[Tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or not href.lower().endswith(exts):
            continue
        absolute = urljoin(page_url, href)
        filename = os.path.basename(urlparse(absolute).path) or short_slug(absolute)
        out.append((absolute, filename))
    return out


def write_meta_sidecar(file_path: str, page_url: str, file_url: str) -> None:
    """Drop a ``<file>.meta.json`` next to ``file_path`` with provenance info."""
    meta = {
        "page_url":        page_url,
        "file_url":        file_url,
        "save_path":       file_path,
        "scraped_at":      dt.datetime.now().isoformat(),
        "file_size_bytes": os.path.getsize(file_path),
        "file_extension":  os.path.splitext(file_path)[1].lower(),
        "folder_tags":     page_url.split("/")[2:],
    }
    with open(file_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------- fetch
def fetch_one_file(
    page_url: str,
    file_url: str,
    filename: str,
    out_dir:  str,
    session:  requests.Session,
    cfg:      HarvestConfig,
) -> PipelineRecord:
    """Stream one file → ``PipelineRecord``. Status in ``{ok, skipped, failed}``."""
    ts        = dt.datetime.now().isoformat()
    save_path = os.path.join(out_dir, sanitize_path(filename))

    # Path-length pre-emptive fallback (Windows MAX_PATH ~260)
    if len(save_path) > 240:
        ext       = os.path.splitext(filename)[1]
        save_path = os.path.join(out_dir, short_slug(file_url, ext=ext))

    if not cfg.overwrite and os.path.exists(save_path):
        return PipelineRecord.skipped(
            id=file_url, reason="already_exists",
            data={"page_url": page_url, "file_path": save_path, "scraped_at": ts},
        )

    if cfg.request_delay:
        time.sleep(cfg.request_delay)

    exc:  Exception | None             = None
    resp: requests.Response | None     = None
    for attempt in range(cfg.max_retries + 1):
        try:
            resp = session.get(file_url, timeout=cfg.timeout, stream=True)
        except requests.exceptions.RequestException as e:
            exc = e
            break
        if resp.status_code == 429 and attempt < cfg.max_retries:
            time.sleep(_retry_delay(resp.headers, attempt))
            continue
        break

    if exc is not None:
        return PipelineRecord.failed(
            id=file_url, reason=f"request_error: {exc}",
            data={"page_url": page_url, "scraped_at": ts},
        )
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return PipelineRecord.failed(
            id=file_url, reason=f"request_error: {e}",
            data={"page_url": page_url, "status_code": resp.status_code, "scraped_at": ts},
        )

    total = int(resp.headers.get("Content-Length", 0))
    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(save_path, "wb") as f, tqdm(
            desc=os.path.basename(save_path),
            total=total or None,
            unit="B", unit_scale=True, unit_divisor=1024,
            disable=not sys.stderr.isatty(),
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=cfg.chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    except OSError:
        # Filesystem rejected the path (length / charset). Retry hash-only short name.
        ext       = os.path.splitext(filename)[1]
        save_path = os.path.join(out_dir, short_slug(file_url, ext=ext))
        try:
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=cfg.chunk_size):
                    if chunk:
                        f.write(chunk)
        except OSError as e2:
            return PipelineRecord.failed(
                id=file_url, reason=f"write_error: {e2}",
                data={"page_url": page_url, "scraped_at": ts},
            )

    write_meta_sidecar(save_path, page_url, file_url)
    return PipelineRecord.ok(
        id=file_url,
        data={
            "page_url":       page_url,
            "file_path":      save_path,
            "content_length": os.path.getsize(save_path),
            "scraped_at":     ts,
        },
    )


# ----------------------------------------------------------------- harvest one
def harvest_one_page(
    page_url: str,
    cfg:      HarvestConfig,
    session:  requests.Session,
    logger:   logging.Logger,
) -> List[PipelineRecord]:
    """Process one listing page → list of records (one per matching file, plus a
    page-level skipped/failed record if the listing fetch itself failed)."""
    out_dir = os.path.join(
        cfg.output_dir,
        sanitize_path(Path(page_url).name) or short_slug(page_url),
    )
    os.makedirs(out_dir, exist_ok=True)

    try:
        files = list_files_on_page(session, page_url, cfg.extensions, cfg.timeout)
    except requests.exceptions.RequestException as e:
        logger.error(f"failed to read listing {page_url}: {e}")
        return [PipelineRecord.failed(
            id=page_url, reason=f"listing_error: {e}",
            data={"scraped_at": dt.datetime.now().isoformat()},
        )]

    if not files:
        logger.warning(f"no matching links on {page_url}")
        return [PipelineRecord.skipped(
            id=page_url, reason="no_matching_links",
            data={"scraped_at": dt.datetime.now().isoformat()},
        )]

    return [fetch_one_file(page_url, fu, fn, out_dir, session, cfg) for fu, fn in files]


def harvest(
    pages: List[str], cfg: HarvestConfig, logger: logging.Logger,
) -> List[PipelineRecord]:
    """Process all listing pages in parallel; flatten to a single record list."""
    session = setup_session(cfg.auth_type, cfg.verify_ssl)
    records: List[PipelineRecord] = []
    pbar = tqdm(total=len(pages), desc="harvest", disable=not sys.stderr.isatty())
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        futures = [ex.submit(harvest_one_page, p, cfg, session, logger) for p in pages]
        for fut in futures:
            try:
                records.extend(fut.result())
            except Exception as e:
                logger.error(f"thread failed: {e}")
            pbar.update(1)
    pbar.close()
    return records


# -------------------------------------------------------------------- runner
def run(args: argparse.Namespace, logger: logging.Logger) -> None:
    pages = load_pages(args.input, logger)
    if not pages:
        utils.save_jsonl([], args.output, logger)
        return

    cfg = HarvestConfig(
        extensions    = [e.strip().lstrip(".") for e in args.extensions.split(",") if e.strip()],
        auth_type     = args.auth,
        output_dir    = args.output_dir,
        max_workers   = args.max_workers,
        timeout       = args.timeout,
        chunk_size    = args.chunk_size,
        verify_ssl    = args.verify_ssl,
        overwrite     = args.overwrite,
        request_delay = args.request_delay,
        max_retries   = args.max_retries,
    )

    if args.dry_run:
        print(f"[dry-run] pages={len(pages)} extensions={cfg.extensions} auth={cfg.auth_type} → {cfg.output_dir}")
        return

    utils.ensure_output_dir()
    os.makedirs(cfg.output_dir, exist_ok=True)
    records = harvest(pages, cfg, logger)
    utils.save_jsonl([record_to_dict(r) for r in records], args.output, logger)
    utils.write_report(records, str(params.REPORT_PATH), header_lines=["=== harvest report ==="])
    logger.info(f"Report: {params.REPORT_PATH}")

    failed = sum(1 for r in records if r.status == "failed")
    if failed:
        logger.warning(f"{failed} item(s) failed — see {params.REPORT_PATH}")
    print(f"Output: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Harvest matching files from listing pages (intranet-aware).",
    )
    p.add_argument("input", nargs="?", default=params.URLS_PATH,
                   help="listing-pages.txt (one URL per line; '#' comments)")
    p.add_argument("-o", "--output", default=params.HARVEST_SUMMARY_PATH,
                   help="output JSONL")
    p.add_argument("--extensions", default="pdf,docx",
                   help="comma-separated file extensions to harvest (default: %(default)s)")
    p.add_argument("--auth", choices=["none", "basic", "ntlm"], default="none",
                   help="HTTP auth type (creds via WEB_READER_USER/_PASSWORD env)")
    p.add_argument("--output-dir", default=params.FILES_DIR,
                   help=f"where to save files (per-URL subdir; default: {params.FILES_DIR})")
    p.add_argument("--max-workers",  type=int,   default=params.MAX_WORKERS)
    p.add_argument("--timeout",      type=int,   default=params.REQUEST_TIMEOUT)
    p.add_argument("--chunk-size",   type=int,   default=8192,
                   help="streaming download chunk size in bytes")
    p.add_argument("--no-verify-ssl", dest="verify_ssl", action="store_false", default=True,
                   help="disable TLS verification (for intranet self-signed certs)")
    p.add_argument("--overwrite",    action="store_true", default=params.OVERWRITE)
    p.add_argument("--request-delay", type=float, default=params.REQUEST_DELAY)
    p.add_argument("--max-retries",   type=int,   default=params.MAX_RETRIES)
    p.add_argument("--dry-run", action="store_true", help="preview without fetching or writing")
    p.add_argument("--verbose", action="store_true", help="enable DEBUG logging")
    return p


def main() -> None:
    """Console entry point: parse args, configure logging, run the harvester."""
    args = build_parser().parse_args()
    logger = utils.setup_logger(params.LOG_FILE_PATH, stream=False)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    run(args, logger)


if __name__ == "__main__":
    main()
