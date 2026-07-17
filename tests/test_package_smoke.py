"""Smoke tests for the top-level :mod:`dstoolbox` package."""

from __future__ import annotations

import pytest


def test_import_top_level():
    import dstoolbox

    assert dstoolbox.__version__ == "0.4.0"


def test_utils_public_surface():
    from dstoolbox import utils

    for name in (
        "flatten_list",
        "regex_filter_list",
        "movecol",
        "reduce_mem_usage",
        "custom_print",
    ):
        assert hasattr(utils, name), f"utils missing {name}"


def test_common_funcs_alias_removed():
    with pytest.raises(ModuleNotFoundError):
        import dstoolbox.common_funcs  # noqa: F401


def test_logging_config_module_available():
    from dstoolbox.logging_config import configure_logging, get_logger

    logger = configure_logging()
    assert logger.name == "dstoolbox"
    assert get_logger("io_funcs.mssql").name == "dstoolbox.io_funcs.mssql"


def test_typed_exceptions_importable():
    from dstoolbox.io_funcs.exceptions import (
        BlobError,
        MSSQLError,
        OutputSpecError,
    )
    from dstoolbox.utils.exceptions import (
        InvalidConfigError,
        OutputFolderError,
    )

    assert issubclass(BlobError, Exception)
    assert issubclass(MSSQLError, Exception)
    assert issubclass(OutputSpecError, Exception)
    assert issubclass(InvalidConfigError, Exception)
    assert issubclass(OutputFolderError, Exception)


def test_ml_funcs_exceptions_importable():
    from dstoolbox.ml_funcs.exceptions import MLFuncsError

    assert issubclass(MLFuncsError, Exception)


def test_custom_print_emits_deprecation_warning():
    from dstoolbox.utils import custom_print
    from dstoolbox.utils.logging_utils import _DEPRECATION_EMITTED  # noqa: F401
    import warnings
    import dstoolbox.utils.logging_utils as lu

    # Reset the module-level guard so this test is order-independent.
    lu._DEPRECATION_EMITTED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        custom_print("hello")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
