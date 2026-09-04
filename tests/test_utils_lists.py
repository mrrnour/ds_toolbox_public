"""Tests for :mod:`dstoolbox.utils.lists`."""

from __future__ import annotations

from dstoolbox.utils.lists import (
    flatten_list,
    regex_filter_list,
    remove_extra_none,
    unique_list,
)


class TestFlattenList:
    def test_flattens_one_level(self):
        assert flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_empty_input_returns_empty(self):
        assert flatten_list([]) == []

    def test_empty_inner_lists(self):
        assert flatten_list([[], [1], []]) == [1]

    def test_preserves_order(self):
        assert flatten_list([["b"], ["a"], ["c"]]) == ["b", "a", "c"]

    def test_scalars_pass_through(self):
        assert flatten_list([1, [2, 3], 4]) == [1, 2, 3, 4]


class TestInWithReg:
    def test_matches_prefix_returns_matched_subset_and_bool_mask(self):
        matches, ind = regex_filter_list([r"^ap"], ["apple", "banana", "apricot"])
        assert sorted(matches) == ["apple", "apricot"]
        assert list(ind) == [True, False, True]

    def test_wraps_single_string_input(self):
        matches, ind = regex_filter_list(r"^b", ["apple", "banana"])
        assert matches == ["banana"]
        assert list(ind) == [False, True]

    def test_no_matches_yields_all_false_mask(self):
        matches, ind = regex_filter_list([r"^zz"], ["apple", "banana"])
        assert matches == []
        assert list(ind) == [False, False]


class TestUniqueList:
    def test_preserves_first_occurrence_order(self):
        assert unique_list([1, 2, 1, 3, 2]) == [1, 2, 3]

    def test_empty(self):
        assert unique_list([]) == []


class TestRemoveExtraNone:
    def test_drops_literal_none_string_when_other_values_exist(self):
        assert remove_extra_none(["a", "None", "b"]) == ["a", "b"]

    def test_keeps_none_string_when_it_is_the_only_value(self):
        assert remove_extra_none(["None"]) == ["None"]

    def test_deduplicates_while_preserving_order(self):
        assert remove_extra_none(["a", "b", "a", "c"]) == ["a", "b", "c"]
