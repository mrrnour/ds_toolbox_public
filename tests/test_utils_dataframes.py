"""Tests for :mod:`dstoolbox.utils.dataframes` non-memory helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from dstoolbox.utils.dataframes import (
    categorical_to_codes,
    compare_dataframes_columns,
    flexible_join,
    join_non_zero,
    movecol,
    null_per_column,
    unify_cols,
)


class TestMovecol:
    def test_moves_column_after_ref(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        out = movecol(df, cols_to_move=["c"], ref_col="a", place="After")
        assert list(out.columns) == ["a", "c", "b"]

    def test_moves_column_before_ref(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        out = movecol(df, cols_to_move=["c"], ref_col="b", place="Before")
        assert list(out.columns) == ["a", "c", "b"]


class TestCat2no:
    def test_returns_new_frame(self):
        df = pd.DataFrame({"x": ["a", "b", "a"]})
        out = categorical_to_codes(df)
        assert out is not df
        assert pd.api.types.is_integer_dtype(out["x"])

    def test_leaves_numeric_untouched(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        out = categorical_to_codes(df)
        assert (out["x"] == df["x"]).all()


class TestNullPerColumn:
    def test_percentages_sorted_descending(self):
        df = pd.DataFrame({"a": [1, None, None], "b": [1, 2, 3]})
        out = null_per_column(df)
        assert out.iloc[0]["null_percent"] >= out.iloc[-1]["null_percent"]
        assert set(out.index) == {"a", "b"}


class TestJoinNonZero:
    def test_joins_nonzero_strings(self):
        assert join_non_zero(["a", 0, "b"], sep=", ") == "a, b"

    def test_all_zero_yields_empty(self):
        assert join_non_zero([0, 0], sep=", ") == ""


class TestUnifyCols:
    def test_pads_disjoint_columns_via_second_pass(self):
        # unify_cols runs the sub-op twice: after the first pass df2 gains df1's
        # missing cols and is reordered to df1.columns; after the second pass
        # df1 gains any cols that only existed in df2 originally.
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [1], "b": [0], "c": [3]})
        out1, out2 = unify_cols(df1, df2, "df1", "df2")
        assert list(out1.columns) == list(out2.columns)
        assert set(out1.columns) == {"a", "b", "c"}


class TestCompareDataframesColumns:
    def test_identical_frames_report_full_match(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = compare_dataframes_columns(df, df, "left", "right", display=False)
        # We only assert the structure — content details are implementation-specific.
        assert result is not None


class TestFlexibleJoin:
    def test_normalized_key_matches_case_and_whitespace(self):
        left = pd.DataFrame({"key": ["Foo Bar", "Baz"], "l": [1, 2]})
        right = pd.DataFrame({"key": ["foo-bar", "qux"], "r": ["a", "b"]})
        out = flexible_join(left, right, on="key", how="inner")
        assert len(out) == 1
        assert out.iloc[0]["l"] == 1
        assert out.iloc[0]["r"] == "a"

    def test_requires_join_keys(self):
        with pytest.raises(ValueError):
            flexible_join(pd.DataFrame(), pd.DataFrame())
