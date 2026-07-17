"""Shared utilities: logging, PipelineRecord, JSONL I/O, output dir helper."""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from . import params


@dataclass
class PipelineRecord:
    """Per-URL unit of work between stages. See CONTEXT.md for status/reason/cause semantics."""

    id: str
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    cause: str | None = None

    @classmethod
    def ok(cls, id: str, data: dict[str, Any] | None = None) -> "PipelineRecord":
        return cls(id=id, status="ok", data=data or {})

    @classmethod
    def skipped(
        cls,
        id: str,
        reason: str,
        data: dict[str, Any] | None = None,
        cause: str | None = None,
    ) -> "PipelineRecord":
        return cls(id=id, status="skipped", data=data or {}, reason=reason, cause=cause)

    @classmethod
    def failed(
        cls,
        id: str,
        reason: str,
        data: dict[str, Any] | None = None,
        cause: str | None = None,
    ) -> "PipelineRecord":
        return cls(id=id, status="failed", data=data or {}, reason=reason, cause=cause)


def record_to_dict(r: PipelineRecord) -> dict[str, Any]:
    return asdict(r)


def dict_to_record(d: dict[str, Any]) -> PipelineRecord:
    return PipelineRecord(
        id=d["id"],
        status=d["status"],
        data=d.get("data", {}),
        reason=d.get("reason") or d.get("error"),
        cause=d.get("cause"),
    )


def ensure_output_dir() -> None:
    os.makedirs(params.OUTPUT_DIR, exist_ok=True)


def setup_logger(log_path: str, *, stream: bool = True) -> logging.Logger:
    """File logger. Set stream=False when using tqdm to avoid clobbering the progress bar."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("web_reader")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def save_jsonl(records: Iterable[dict[str, Any]], path: str, logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1
    logger.info(f"Wrote {count} records to {path}")


def load_jsonl(path: str, logger: logging.Logger) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        sys.exit(f"Error: input not found: {path}")
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            records.append(json.loads(ln))
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def _format_record_line(r: PipelineRecord) -> str:
    parts = [f"  {r.id}", f"reason={r.reason}"]
    if r.cause:
        parts.append(f"cause={r.cause}")
    sc = r.data.get("status_code") if r.data else None
    if sc:
        parts.append(f"status_code={sc}")
    return "  ".join(parts)


def write_report(
    records: list[PipelineRecord],
    path: str,
    header_lines: list[str] | None = None,
) -> None:
    """Render a uniform per-stage report. Caller supplies stage-specific header lines."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = sum(1 for r in records if r.status == "ok")
    skipped = [r for r in records if r.status == "skipped"]
    failed = [r for r in records if r.status == "failed"]
    lines: list[str] = list(header_lines or [])
    lines += [
        f"total:    {len(records)}",
        f"ok:       {ok}",
        f"skipped:  {len(skipped)}",
        f"failed:   {len(failed)}",
        "",
    ]
    if skipped:
        lines.append("--- skipped ---")
        lines.extend(_format_record_line(r) for r in skipped)
        lines.append("")
    if failed:
        lines.append("--- failed ---")
        lines.extend(_format_record_line(r) for r in failed)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def check_input_size(path: str) -> None:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > params.MAX_INPUT_MB:
        sys.exit(
            f"Error: input file {size_mb:.1f}MB exceeds limit of {params.MAX_INPUT_MB}MB"
        )
