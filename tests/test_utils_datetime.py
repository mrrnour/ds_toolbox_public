"""Tests for :mod:`dstoolbox.utils.datetime_utils`."""

from __future__ import annotations

import pytest

from dstoolbox.utils.datetime_utils import extract_start_end


class TestExtractStartEnd:
    def test_end_is_the_day_before_the_next_boundary(self):
        udates = ["2021-01-01", "2021-02-01", "2021-03-01"]
        assert extract_start_end(udates, 0) == ("2021-01-01", "2021-01-31")

    def test_windows_are_contiguous_but_do_not_overlap(self):
        udates = ["2021-01-01", "2021-02-01", "2021-03-01"]
        _, first_end = extract_start_end(udates, 0)
        second_start, _ = extract_start_end(udates, 1)
        assert first_end < second_start

    def test_handles_month_and_year_rollover(self):
        assert extract_start_end(["2020-12-01", "2021-01-01"], 0) == ("2020-12-01", "2020-12-31")

    def test_leap_day_is_preserved(self):
        assert extract_start_end(["2020-02-01", "2020-03-01"], 0) == ("2020-02-01", "2020-02-29")

    def test_start_is_returned_verbatim(self):
        udates = ["2021-04-09", "2021-04-22"]
        assert extract_start_end(udates, 0) == ("2021-04-09", "2021-04-21")

    def test_last_index_has_no_following_boundary(self):
        with pytest.raises(IndexError):
            extract_start_end(["2021-01-01", "2021-02-01"], 1)
