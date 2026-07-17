"""Helper for forecaster adapters with heavy optional dependencies.

Each backend (statsforecast, greykite, darts, ...) is imported lazily at
adapter-import time using :func:`optional_import`. If the dependency is
missing, an instructive error is raised *only when the adapter is used*,
not when the package is imported.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


class _MissingDependency:
    """Placeholder that raises a helpful error on any attribute access."""

    def __init__(self, module_name: str, used_by: str, install_hint: str) -> None:
        self._module_name = module_name
        self._used_by = used_by
        self._install_hint = install_hint

    def __getattr__(self, item: str) -> Any:
        raise ImportError(
            f"{self._used_by} requires {self._module_name!r}, which is not installed. "
            f"Install it with: {self._install_hint}"
        )

    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"{self._used_by} requires {self._module_name!r}, which is not installed. "
            f"Install it with: {self._install_hint}"
        )


_HINTS = {
    "statsforecast": "pip install statsforecast",
    "statsforecast.models": "pip install statsforecast",
    "greykite.framework.templates.forecaster": "pip install greykite",
    "greykite.framework.templates.autogen.forecast_config": "pip install greykite",
    "greykite.common.constants": "pip install greykite",
    "darts.timeseries": "pip install darts",
    "darts.models": "pip install 'darts[torch]'",
    "darts.utils.utils": "pip install darts",
}


def optional_import(
    module_name: str,
    used_by: str,
    required: bool = True,
) -> ModuleType | _MissingDependency:
    """Import ``module_name`` if available; otherwise return a lazy error stub.

    Parameters
    ----------
    module_name
        Fully-qualified module path.
    used_by
        Name of the adapter / class that needs this module. Surfaced in the
        error message when the missing dep is accessed.
    required
        If ``False`` and the module is missing, ``None`` is returned instead
        of an error stub (lets adapters opt into truly soft dependencies).
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if not required:
            return None  # type: ignore[return-value]
        hint = _HINTS.get(module_name, f"pip install {module_name.split('.')[0]}")
        return _MissingDependency(module_name, used_by, hint)
