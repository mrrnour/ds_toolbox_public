"""Tests for the ``dsbayes-group`` entry point.

Argument parsing and failure handling are checked without fitting anything.
Only the two end-to-end tests sample, and they use a tiny chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dstoolbox.ml_funcs.stat_bayes_group_cli import _window_arg, main

PRE = "2026-01-01:2026-01-10"
POST = "2026-01-11:2026-01-20"
SMALL = ["--draws", "300", "--tune", "300", "--chains", "2", "--quiet"]


@pytest.fixture
def events_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for start, end, rate in (
        ("2026-01-01", "2026-01-10", 0.10),
        ("2026-01-11", "2026-01-20", 0.25),
    ):
        days = pd.date_range(start, end)
        for user in range(150):
            for day in rng.choice(days, size=rng.integers(1, 4), replace=False):
                rows.append((user, str(day)[:10], int(rng.random() < rate)))
    path = tmp_path / "events.csv"
    pd.DataFrame(rows, columns=["user_id", "datepart", "convert"]).to_csv(path, index=False)
    return str(path)


# --------------------------------------------------------------------------- #
# Argument parsing — no fitting
# --------------------------------------------------------------------------- #


def test_window_arg_splits_on_the_colon():
    assert _window_arg("2026-01-01:2026-01-10") == ("2026-01-01", "2026-01-10")


@pytest.mark.parametrize("text", ["2026-01-01", ":2026-01-10", "2026-01-01:", ""])
def test_window_arg_rejects_malformed_ranges(text):
    with pytest.raises(Exception, match="START:END"):
        _window_arg(text)


def test_half_a_window_is_an_error(capsys):
    """``--pre`` without ``--post`` is a half-specified split, not a default."""
    assert main(["events.csv", "--pre", PRE]) == 1
    assert "--pre and --post go together" in capsys.readouterr().err


def test_a_lone_csv_with_no_split_is_an_error(capsys):
    """One file and no rule for dividing it cannot name two groups."""
    assert main(["events.csv"]) == 1
    assert "needs a split" in capsys.readouterr().err


def test_two_files_plus_a_split_flag_is_ambiguous(capsys):
    """Two files already define the groups; a second rule would contradict them."""
    assert main(["a.csv", "b.csv", "--group-col", "arm"]) == 1
    assert "drop --group-col" in capsys.readouterr().err


def test_group_col_and_dates_are_two_different_splits(capsys):
    assert main(["events.csv", "--group-col", "arm", "--pre", PRE, "--post", POST]) == 1
    assert "pick one" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Failures exit 1 with a message, never a traceback
# --------------------------------------------------------------------------- #


def test_missing_file_exits_nonzero(capsys):
    code = main(["nope.csv", "--pre", PRE, "--post", POST, *SMALL])
    assert code == 1
    assert "nope.csv" in capsys.readouterr().err


def test_wrong_column_name_exits_nonzero(events_csv, capsys):
    code = main([events_csv, "--pre", PRE, "--post", POST, "--metric-col", "clicked", *SMALL])
    assert code == 1
    assert capsys.readouterr().err.strip()


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_quiet_prints_only_the_verdict(events_csv, capsys):
    pytest.importorskip("pymc")
    code = main([events_csv, "--pre", PRE, "--post", POST, "--rope-pct", "0.10", *SMALL])
    out = capsys.readouterr().out.strip()
    assert code == 0
    assert out == "positive"


def test_writes_csv_and_plot(events_csv, tmp_path, capsys):
    pytest.importorskip("pymc")
    csv_out, png_out = tmp_path / "row.csv", tmp_path / "fig.png"
    code = main(
        [
            events_csv,
            "--pre",
            PRE,
            "--post",
            POST,
            "--rope-pct",
            "0.10",
            "--csv",
            str(csv_out),
            "--plot",
            str(png_out),
            *SMALL,
        ]
    )
    capsys.readouterr()

    assert code == 0
    assert png_out.stat().st_size > 0
    row = pd.read_csv(csv_out)
    assert len(row) == 1
    assert row.loc[0, "decision"] == "positive"
    assert row.loc[0, "n_days_pre"] == row.loc[0, "n_days_post"] == 10
