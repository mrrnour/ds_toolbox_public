"""AST guard: no library module under ``dstoolbox`` may call ``sys.exit``.

Web-reader CLIs are the only sanctioned exception (they are entry-point
scripts, not library code).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "dstoolbox"
ALLOWED_SUBDIRS = {"web_reader", "doc"}


def _library_py_files():
    for path in PACKAGE_ROOT.rglob("*.py"):
        rel = path.relative_to(PACKAGE_ROOT)
        if rel.parts and rel.parts[0] in ALLOWED_SUBDIRS:
            continue
        yield path


class _SysExitFinder(ast.NodeVisitor):
    def __init__(self):
        self.hits: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "exit":
            value = func.value
            if isinstance(value, ast.Name) and value.id == "sys":
                self.hits.append((node.lineno, "sys.exit(...)"))
        self.generic_visit(node)


@pytest.mark.parametrize("path", list(_library_py_files()), ids=lambda p: str(p.relative_to(PACKAGE_ROOT)))
def test_no_sys_exit_in_library_module(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    finder = _SysExitFinder()
    finder.visit(tree)
    assert not finder.hits, (
        f"{path} contains disallowed sys.exit call(s): {finder.hits}. "
        "Raise a typed exception from dstoolbox.<subpkg>.exceptions instead."
    )
