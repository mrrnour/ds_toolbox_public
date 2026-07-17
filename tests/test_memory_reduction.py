"""Tests for the memory-reduction plan/apply split.

Covers the pure dtype-selection logic in ``_plan_memory_reduction`` and the
non-mutating contract of ``_apply_memory_reduction``.
"""

import numpy as np
import pandas as pd
import pytest

from dstoolbox.utils.dataframes import (
    _apply_memory_reduction,
    _plan_memory_reduction,
    reduce_mem_usage,
)


def _targets(plan: pd.DataFrame) -> dict:
    """Column -> target dtype; NaN (missing) is normalized to ``None``."""
    return {c: (None if pd.isna(t) else t)
            for c, t in zip(plan["column"], plan["target_dtype"])}


# ---------- Integer selection ----------

def test_int_fits_int8():
    df = pd.DataFrame({"x": pd.array([-5, 0, 127], dtype="int64")})
    plan = _plan_memory_reduction(df)
    assert _targets(plan) == {"x": "int8"}


def test_int_fits_int16():
    df = pd.DataFrame({"x": pd.array([0, 200, 30_000], dtype="int64")})
    plan = _plan_memory_reduction(df)
    assert _targets(plan) == {"x": "int16"}


def test_int_fits_int32():
    df = pd.DataFrame({"x": pd.array([0, 1_000_000], dtype="int64")})
    plan = _plan_memory_reduction(df)
    assert _targets(plan) == {"x": "int32"}


def test_int_too_big_stays():
    df = pd.DataFrame({"x": pd.array([0, 10**12], dtype="int64")})
    plan = _plan_memory_reduction(df)
    assert _targets(plan) == {"x": None}


# ---------- Float selection ----------

def test_float_finite_picks_float32():
    df = pd.DataFrame({"x": [0.1, 1.5, -3.7]})
    plan = _plan_memory_reduction(df)
    assert _targets(plan) == {"x": "float32"}


def test_float_with_inf_left_alone():
    df = pd.DataFrame({"x": [0.1, np.inf, 1.5]})
    plan = _plan_memory_reduction(df)
    assert _targets(plan) == {"x": None}


def test_float_with_nan_left_alone():
    df = pd.DataFrame({"x": [0.1, np.nan, 1.5]})
    plan = _plan_memory_reduction(df)
    assert _targets(plan) == {"x": None}


def test_float16_opt_in():
    df = pd.DataFrame({"x": [0.0, 1.0, 100.0]})
    plan = _plan_memory_reduction(df, use_float16=True)
    assert _targets(plan) == {"x": "float16"}


def test_float16_off_by_default():
    df = pd.DataFrame({"x": [0.0, 1.0, 100.0]})
    plan = _plan_memory_reduction(df)
    assert _targets(plan) == {"x": "float32"}


# ---------- Object / string / category ----------

def test_object_to_category_by_default():
    df = pd.DataFrame({"x": ["a", "b", "a"]})
    plan = _plan_memory_reduction(df)
    assert _targets(plan) == {"x": "category"}


def test_object_to_string_when_str2cat_empty():
    # Force object dtype: pandas 3 infers `str` for plain string literals.
    df = pd.DataFrame({"x": pd.Series(["a", "b", "a"], dtype=object)})
    plan = _plan_memory_reduction(df, str2cat_cols=[])
    assert _targets(plan) == {"x": "string"}


def test_object_left_alone_when_neither_selected():
    df = pd.DataFrame({"x": ["a", "b", "a"]})
    plan = _plan_memory_reduction(df, obj2str_cols=[], str2cat_cols=[])
    assert _targets(plan) == {"x": None}


# ---------- Datetime skipped ----------

def test_datetime_left_alone():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    plan = _plan_memory_reduction(df)
    assert _targets(plan) == {"d": None}
    assert "datetime" in plan.iloc[0]["reason"].lower()


# ---------- Mixed frame ----------

def test_mixed_frame_full_decisions():
    df = pd.DataFrame({
        "small_int": pd.array([1, 2, 3], dtype="int64"),
        "big_int":   pd.array([1, 10**12], dtype="int64").repeat(2)[:3],
        "f":         [0.5, 1.5, 2.5],
        "label":     ["a", "b", "a"],
        "when":      pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })
    plan = _plan_memory_reduction(df)
    targets = _targets(plan)
    assert targets["small_int"] == "int8"
    assert targets["big_int"] is None
    assert targets["f"] == "float32"
    assert targets["label"] == "category"
    assert targets["when"] is None


# ---------- Apply contract ----------

def test_apply_does_not_mutate_input():
    df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype="int64")})
    original_dtype = df["x"].dtype
    plan = _plan_memory_reduction(df)
    out = _apply_memory_reduction(df, plan)
    assert df["x"].dtype == original_dtype, "input was mutated"
    assert str(out["x"].dtype) == "int8"


def test_apply_with_empty_plan_is_identity():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    plan = pd.DataFrame(columns=["column", "original_dtype", "target_dtype", "reason"])
    out = _apply_memory_reduction(df, plan)
    pd.testing.assert_frame_equal(out, df)


# ---------- Convenience wrapper ----------

def test_reduce_mem_usage_returns_new_frame_and_shrinks(capsys):
    df = pd.DataFrame({
        "x": pd.array([1, 2, 3, 4], dtype="int64"),
        "y": [0.1, 0.2, 0.3, 0.4],
    })
    before_dtypes = df.dtypes.copy()
    out = reduce_mem_usage(df)
    # Input not mutated
    pd.testing.assert_series_equal(df.dtypes, before_dtypes)
    # Output is smaller
    assert out["x"].dtype == np.dtype("int8")
    assert out["y"].dtype == np.dtype("float32")
    # Memory summary printed (not asserting exact bytes — just that it ran)
    captured = capsys.readouterr().out
    assert "Memory usage of dataframe" in captured
    assert "Decreased by" in captured


# ---------- Plan.changes filter ----------

def test_plan_changes_filters_to_actual_dtype_changes():
    df = pd.DataFrame({
        "small":   pd.array([1, 2], dtype="int64"),
        "skipped": pd.to_datetime(["2024-01-01", "2024-01-02"]),
    })
    plan = _plan_memory_reduction(df)
    assert len(plan) == 2
    changes = plan[plan["target_dtype"].notna()]
    assert len(changes) == 1
    assert changes.iloc[0]["column"] == "small"
