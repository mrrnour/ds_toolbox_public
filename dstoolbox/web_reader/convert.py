#!/usr/bin/env python3
"""Convert raw HTML (from crawling_summary.jsonl) into clean Markdown files."""

import argparse
import datetime as dt
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from markdownify import markdownify as md_from_html
from pypdf import PdfReader
from tqdm import tqdm

load_dotenv()

import contextlib

from . import params, utils
from .utils import PipelineRecord, dict_to_record, record_to_dict


@dataclass
class ConvertConfig:
    input_path: str = field(default_factory=lambda: params.CRAWLING_SUMMARY_PATH)
    output_dir: str = field(default_factory=lambda: params.MARKDOWN_DIR)
    output_path: str = field(default_factory=lambda: params.CONVERSION_SUMMARY_PATH)
    min_length: int = field(default_factory=lambda: params.MIN_MARKDOWN_LENGTH)
    folder_name: str = ""  # Optional bookmark folder name for organizing output
    overwrite: bool = True
    dry_run: bool = False
    verbose: bool = False
    use_docling: bool = False
    use_images: bool = False


BOILERPLATE_TAGS = ("script", "style", "nav", "header", "footer", "aside")

DOCLING_EXTENSIONS = frozenset(
    {
        ".html",
        ".htm",
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".md",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".bmp",
    }
)


def setup_docling(use_images: bool = False):
    """Build a DocumentConverter. Lazy-imports docling so it's optional."""
    logging.getLogger("docling").setLevel(logging.ERROR)
    try:
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import (
            DocumentConverter,
            InputFormat,
            PdfFormatOption,
            StandardPdfPipeline,
        )
    except ImportError:
        sys.exit("docling not installed — run: pip install docling")

    opts = PdfPipelineOptions()
    opts.do_ocr = use_images
    opts.do_table_structure = True
    opts.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)

    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.XLSX,
            InputFormat.MD,
            InputFormat.IMAGE,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=opts,
            ),
        },
    )


def docling_to_markdown(
    file_path: str, doc_converter, min_length: int
) -> tuple[str | None, str | None]:
    """Convert any docling-supported file to markdown. Returns (markdown, skip_reason)."""
    try:
        result = doc_converter.convert(file_path)
    except Exception as e:
        return None, f"docling_error: {e}"
    if result.document is None:
        return None, "docling_empty_result"
    md = result.document.export_to_markdown().strip()
    if len(md) < min_length:
        return None, "below_min_length"
    return md, None


def strip_boilerplate(soup: BeautifulSoup) -> BeautifulSoup:
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()
    return soup


def html_to_markdown(html: str, min_length: int) -> tuple[str | None, str | None]:
    """Return (markdown, skip_reason). markdown is None iff content is below min_length."""
    soup = strip_boilerplate(BeautifulSoup(html, "html.parser"))
    md = md_from_html(str(soup), heading_style="ATX", strip=["img"]).strip()
    if len(soup.get_text(" ", strip=True)) < min_length:
        return None, "below_min_length"
    return md, None


def pdf_to_markdown(pdf_path: str, min_length: int) -> tuple[str | None, str | None]:
    """Extract text per page, join with '## Page N' separators."""
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        return None, f"pdf_read_error: {e}"
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            parts.append(f"## Page {i}\n\n{text}")
    if not parts:
        return None, "pdf_no_text_extracted"
    md = "\n\n".join(parts).strip()
    if len(md) < min_length:
        return None, "below_min_length"
    return md, None


def build_frontmatter(url: str, scraped_at: str, depth: int) -> str:
    return f"---\nurl: {url}\nscraped_at: {scraped_at}\ndepth: {depth}\n---\n\n"


def slugify(text: str) -> str:
    """Convert text to kebab-case slug (lowercase, hyphens, alphanumeric only)."""
    import re

    # Convert to lowercase and replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    # Remove multiple consecutive hyphens
    slug = re.sub(r"-+", "-", slug)
    # Limit to 50 chars to keep filenames reasonable
    return slug[:50]


def unique_path_with_numeric_prefix(path: str) -> str:
    """Return a non-existing path by prefixing N- when needed.

    Example: file.md -> 2-file.md -> 3-file.md
    """
    if not os.path.exists(path):
        return path

    parent = os.path.dirname(path)
    name = os.path.basename(path)
    idx = 2
    while True:
        candidate = os.path.join(parent, f"{idx}-{name}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def stem_from_url(url: str) -> str:
    """Build a readable stem from URL path/netloc when title is unavailable."""
    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    raw = os.path.basename(parsed.path.rstrip("/"))
    raw = os.path.splitext(raw)[0]
    raw = unquote(raw)
    if not raw:
        raw = parsed.netloc or "page"
    stem = slugify(raw)
    return stem or "page"


def normalize_folder_path(path: str) -> str:
    """Normalize folder paths to slash-separated slug components."""
    if not path:
        return ""
    parts = [slugify(part) for part in path.replace("\\", "/").split("/") if part.strip()]
    parts = [part for part in parts if part]
    return "/".join(parts)


def infer_category_from_url(url: str) -> str:
    """Infer content category from URL domain/path.

    Examples:
        https://towardsdatascience.com/... → "tutorial"
        https://github.com/... → "code"
        https://arxiv.org/... → "paper"
        https://example.com/blog/... → "blog"
        https://example.com/docs/... → "docs"
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    # Domain-based categorization
    if "arxiv.org" in domain or "researchgate.net" in domain:
        return "paper"
    elif "github.com" in domain or "gitlab.com" in domain:
        return "code"
    elif "medium.com" in domain or "towardsdatascience.com" in domain:
        return "tutorial"
    elif "youtube.com" in domain or "youtu.be" in domain:
        return "video"
    elif "wikipedia.org" in domain:
        return "reference"
    elif "stackoverflow.com" in domain:
        return "qa"
    elif "linkedin.com" in domain:
        return "social"

    # Path-based categorization
    if "/blog/" in path or "/news/" in path:
        return "blog"
    elif "/docs/" in path or "/documentation/" in path:
        return "docs"
    elif "/tutorial/" in path or "/guide/" in path:
        return "tutorial"
    elif "/api/" in path or "/reference/" in path:
        return "reference"

    # Default category based on domain
    if "edu" in domain:
        return "academic"
    elif "github" in domain or "gitlab" in domain:
        return "code"
    else:
        return "article"


def extract_title_from_html(html: str) -> str:
    """Extract title from HTML <title> tag or first heading.

    Returns:
        Title text, or empty string if not found.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Try <title> tag first
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        return title_tag.string.strip()

    # Try first <h1>
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    # Try first <h2>
    h2 = soup.find("h2")
    if h2:
        return h2.get_text(strip=True)

    # Try meta og:title
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"]

    return ""


def md_path_for(
    file_path: str, output_dir: str, url: str = "", html: str = "", folder_name: str = ""
) -> str:
    """Generate markdown filename with category prefix and descriptive name.

    Filename format: <category>-<title-slug>.md
    Falls back to URL-based stem (no hash).

    Args:
        file_path: Original file path.
        output_dir: Output directory.
        url: Source URL (for category inference).
        html: HTML content (for title extraction).
        folder_name: Optional bookmark folder name for subdirectory organization.

    Returns:
        Full path to output markdown file.
    """
    # Build the base output directory, adding folder_name as subdirectory if provided
    final_output_dir = output_dir
    if folder_name:
        normalized_folder = normalize_folder_path(folder_name)
        if normalized_folder:
            final_output_dir = os.path.join(output_dir, *normalized_folder.split("/"))

    category = infer_category_from_url(url) if url else "article"
    title = extract_title_from_html(html) if html else ""
    title_slug = slugify(title) if title else ""
    if not title_slug:
        title_slug = stem_from_url(url) if url else "page"

    filename = f"{category}-{title_slug}.md"
    return os.path.join(final_output_dir, filename)


def mirror_html_path_for(md_path: str, folder_name: str = "") -> str:
    """Build an HTML mirror path that matches the markdown stem and folder layout."""
    normalized_folder = normalize_folder_path(folder_name)
    html_root = str(params.HTML_DIR)
    if normalized_folder:
        html_root = os.path.join(html_root, *normalized_folder.split("/"))
    md_stem = os.path.splitext(os.path.basename(md_path))[0]
    return os.path.join(html_root, f"{md_stem}.html")


def cleanup_legacy_hashed_html(html_root: str) -> int:
    """Remove old hashed HTML artifacts like 04ee8b6039_name.html."""
    removed = 0
    pattern = re.compile(r"^[0-9a-f]{10}_.+\.html?$")
    for root, _, files in os.walk(html_root):
        for name in files:
            if pattern.match(name):
                path = os.path.join(root, name)
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
    return removed


def convert_one(
    crawl_rec: PipelineRecord,
    output_dir: str,
    min_length: int,
    overwrite: bool = True,
    doc_converter=None,
    folder_name: str = "",
) -> PipelineRecord:
    ts = dt.datetime.now().isoformat()
    url = crawl_rec.id
    html_content = ""  # Capture HTML for title extraction

    # Get folder_path from crawling record if available (from @folder_path metadata)
    # Otherwise use the folder_name parameter as a fallback
    folder_path_from_rec = crawl_rec.data.get("folder_path", "")
    effective_folder_path = folder_path_from_rec or folder_name

    if crawl_rec.status == "failed":
        return PipelineRecord.skipped(
            id=url,
            reason=f"crawl_{crawl_rec.status}",
            data={"converted_at": ts},
            cause=crawl_rec.reason,
        )

    file_path = crawl_rec.data.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return PipelineRecord.failed(
            id=url,
            reason=f"source_file_missing: {file_path}",
            data={"converted_at": ts},
        )

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if doc_converter is not None:
            if ext not in DOCLING_EXTENSIONS:
                return PipelineRecord.skipped(
                    id=url,
                    reason=f"unsupported_extension: {ext}",
                    data={"converted_at": ts},
                )
            md, skip_reason = docling_to_markdown(file_path, doc_converter, min_length)
        elif ext in (".html", ".htm"):
            with open(file_path, encoding="utf-8") as f:
                html_content = f.read()
            md, skip_reason = html_to_markdown(html_content, min_length)
        elif ext == ".pdf":
            md, skip_reason = pdf_to_markdown(file_path, min_length)
        else:
            return PipelineRecord.skipped(
                id=url,
                reason=f"unsupported_extension: {ext}",
                data={"converted_at": ts},
            )
    except OSError as e:
        return PipelineRecord.failed(
            id=url,
            reason=f"read_error: {e}",
            data={"converted_at": ts},
        )

    if md is None:
        return PipelineRecord.skipped(
            id=url,
            reason=skip_reason,
            data={"converted_at": ts},
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = md_path_for(
        file_path, output_dir, url=url, html=html_content, folder_name=effective_folder_path
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not overwrite and os.path.exists(out_path):
        return PipelineRecord.skipped(
            id=url,
            reason="already_exists",
            data={"converted_at": ts, "md_path": out_path},
        )

    out_path = unique_path_with_numeric_prefix(out_path)

    frontmatter = build_frontmatter(
        url=url,
        scraped_at=crawl_rec.data.get("scraped_at", ""),
        depth=crawl_rec.data.get("depth", 0),
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(frontmatter + md + "\n")

    mirrored_html_path = ""
    if ext in (".html", ".htm"):
        mirrored_html_path = mirror_html_path_for(out_path, effective_folder_path)
        os.makedirs(os.path.dirname(mirrored_html_path), exist_ok=True)
        shutil.copyfile(file_path, mirrored_html_path)
        # Cleanup legacy hashed HTML files so output/html keeps convention names only.
        base_name = os.path.basename(file_path)
        if re.match(r"^[0-9a-f]{10}_.+\.html?$", base_name) and os.path.abspath(
            file_path
        ) != os.path.abspath(mirrored_html_path):
            with contextlib.suppress(OSError):
                os.remove(file_path)

    return PipelineRecord.ok(
        id=url,
        data={
            "md_path": out_path,
            "mirrored_html_path": mirrored_html_path,
            "char_count": len(md),
            "converted_at": ts,
        },
    )


def run(cfg: ConvertConfig, logger: logging.Logger) -> None:
    utils.ensure_output_dir()
    raw = utils.load_jsonl(cfg.input_path, logger)
    crawl_records = [dict_to_record(d) for d in raw]

    if not crawl_records:
        logger.warning("No crawl records to convert")
        utils.save_jsonl([], cfg.output_path, logger)
        return

    if cfg.dry_run:
        print(f"[dry-run] records={len(crawl_records)} → {cfg.output_dir} + {cfg.output_path}")
        return

    doc_converter = setup_docling(cfg.use_images) if cfg.use_docling else None
    os.makedirs(cfg.output_dir, exist_ok=True)
    results: list[PipelineRecord] = []
    for cr in tqdm(crawl_records, desc="convert", disable=not sys.stderr.isatty()):
        results.append(
            convert_one(
                cr, cfg.output_dir, cfg.min_length, cfg.overwrite, doc_converter, cfg.folder_name
            )
        )

    utils.save_jsonl([record_to_dict(r) for r in results], cfg.output_path, logger)
    removed_hashed = cleanup_legacy_hashed_html(str(params.HTML_DIR))
    if removed_hashed:
        logger.info(f"Removed {removed_hashed} legacy hashed HTML file(s)")
    utils.write_report(results, str(params.REPORT_PATH), header_lines=["=== convert report ==="])
    logger.info(f"Report: {params.REPORT_PATH}")
    print(f"Output: {cfg.output_path}")
    print(f"Markdown: {cfg.output_dir}/")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert crawled HTML files to clean Markdown with YAML frontmatter."
    )
    p.add_argument(
        "input", nargs="?", default=params.CRAWLING_SUMMARY_PATH, help="crawling_summary.jsonl"
    )
    p.add_argument("-o", "--output", default=params.CONVERSION_SUMMARY_PATH, help="output JSONL")
    p.add_argument(
        "--output-dir",
        default=params.MARKDOWN_DIR,
        help=f"directory for .md files (default: {params.MARKDOWN_DIR})",
    )
    p.add_argument(
        "--folder", default="", help="bookmark folder name (creates subdirectory in output-dir)"
    )
    p.add_argument(
        "--min-length",
        type=int,
        default=params.MIN_MARKDOWN_LENGTH,
        help="skip pages where extracted text is shorter than N chars",
    )
    p.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help="skip .md files that already exist",
    )
    p.add_argument("--dry-run", action="store_true", help="preview without writing")
    p.add_argument("--verbose", action="store_true", help="enable DEBUG logging")
    p.add_argument(
        "--use-docling",
        action="store_true",
        help="use docling for conversion (supports HTML, PDF, DOCX, PPTX, XLSX, images)",
    )
    p.add_argument(
        "--image",
        action="store_true",
        help="enable OCR for images inside PDFs (requires --use-docling)",
    )
    p.add_argument(
        "--file", default=None, help="convert a single file directly (skips crawling_summary.jsonl)"
    )
    return p


def main() -> None:
    """Console entry point: parse args and run the converter."""
    args = build_parser().parse_args()
    cfg = ConvertConfig(
        input_path=args.input,
        output_path=args.output,
        output_dir=args.output_dir,
        folder_name=args.folder,
        min_length=args.min_length,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verbose=args.verbose,
        use_docling=args.use_docling,
        use_images=args.image,
    )
    logger = utils.setup_logger(params.LOG_FILE_PATH, stream=not sys.stderr.isatty())
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    run(cfg, logger)

    if args.file:
        file_path = os.path.abspath(args.file)
        doc_converter = setup_docling(args.image) if args.use_docling else None
        crawl_rec = PipelineRecord(
            id=file_path,
            status="ok",
            data={"file_path": file_path, "depth": 0, "scraped_at": dt.datetime.now().isoformat()},
        )
        result = convert_one(
            crawl_rec,
            args.output_dir,
            args.min_length,
            overwrite=args.overwrite,
            doc_converter=doc_converter,
        )
        print(f"status: {result.status}")
        if result.status == "ok":
            print(f"output: {result.data['md_path']}")
        else:
            print(f"reason: {result.data.get('skip_reason') or result.error}")
    else:
        cfg = ConvertConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            output_path=args.output,
            min_length=args.min_length,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            verbose=args.verbose,
            use_docling=args.use_docling,
            use_images=args.image,
        )
        run(cfg, logger)


if __name__ == "__main__":
    main()
