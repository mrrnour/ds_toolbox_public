#!/usr/bin/env bash
# Build the dstoolbox API reference with pdoc (mitmproxy/pdoc, not pdoc3).
#
# Uses pdoc's native features to make the site look sharp:
#   --logo / --favicon    → branded sidebar + tab icon
#   --footer-text         → project + license line
#   --math                → MathJax for inline / block equations
#   --mermaid             → Mermaid diagrams inside docstrings
#   --search              → in-page search box (default; kept explicit)
#
# Usage:
#   bash scripts/build_docs.sh            # one-shot build into docs/api/
#   bash scripts/build_docs.sh --serve    # live-reload dev server on :8080
#
# Requires:
#   pip install -e ".[docs]"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="docs/api"

# The top-level dstoolbox/__init__.py lazily imports each subpackage under
# try/except so pdoc can walk whatever imported successfully. Missing extras
# (pyspark, hyperopt, ...) silently drop those subpackages from the docs.
BASE_MODULES=(dstoolbox)
CANDIDATE_SUBPACKAGES=(
    dstoolbox.utils
    dstoolbox.io_funcs
    dstoolbox.ml_funcs
    dstoolbox.spark_funcs
    dstoolbox.nlp_llm_funcs
    dstoolbox.rag_funcs
    dstoolbox.web_reader
)

MODULES=("${BASE_MODULES[@]}")
for m in "${CANDIDATE_SUBPACKAGES[@]}"; do
    if python -c "import $m" 2>/dev/null; then
        MODULES+=("$m")
    else
        echo "note: skipping $m (import failed — missing optional extra?)" >&2
    fi
done

if ! command -v pdoc >/dev/null 2>&1; then
    echo "error: pdoc is not installed. Run: pip install -e \".[docs]\"" >&2
    exit 1
fi

# Branding — override via env vars for CI / GitHub Pages deployments.
#
# The default logo is a base64 data URI of docs/assets/ds_toolbox_logo.png so
# it renders correctly from every nesting depth (e.g. dstoolbox/utils.html)
# without depending on an external CDN or a resolvable relative path.
DEFAULT_LOGO_ASSET="$REPO_ROOT/docs/assets/ds_toolbox_logo_small.png"
if [[ -z "${DSTOOLBOX_DOCS_LOGO_URL:-}" && -f "$DEFAULT_LOGO_ASSET" ]]; then
    DEFAULT_LOGO_DATA_URI="data:image/png;base64,$(base64 < "$DEFAULT_LOGO_ASSET" | tr -d '\n')"
else
    DEFAULT_LOGO_DATA_URI="https://raw.githubusercontent.com/dstoolbox/dstoolbox/main/dstoolbox/images/ds_toolbox_logo.png"
fi

LOGO_URL="${DSTOOLBOX_DOCS_LOGO_URL:-$DEFAULT_LOGO_DATA_URI}"
FAVICON_URL="${DSTOOLBOX_DOCS_FAVICON_URL:-$LOGO_URL}"
FOOTER_TEXT="${DSTOOLBOX_DOCS_FOOTER:-dstoolbox · GPL-3.0-or-later · built with pdoc}"

TEMPLATE_DIR="$REPO_ROOT/docs/templates"

PDOC_COMMON_ARGS=(
    --docformat numpy
    --logo "$LOGO_URL"
    --favicon "$FAVICON_URL"
    --footer-text "$FOOTER_TEXT"
    --math
    --mermaid
    --search
)

if [[ -d "$TEMPLATE_DIR" ]]; then
    PDOC_COMMON_ARGS+=(--template-directory "$TEMPLATE_DIR")
fi

if [[ "${1:-}" == "--serve" ]]; then
    exec pdoc "${PDOC_COMMON_ARGS[@]}" "${MODULES[@]}"
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

pdoc \
    "${PDOC_COMMON_ARGS[@]}" \
    --output-directory "$OUT_DIR" \
    "${MODULES[@]}"

echo "Docs written to $OUT_DIR/index.html"
