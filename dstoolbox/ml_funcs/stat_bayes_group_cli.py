"""``dsbayes-group`` — two-group conversion verdict from the command line.

Give it each group's data and it fits the hierarchical Beta-Binomial and
prints the verdict. Three ways to say which rows belong to which group::

    # two files, one per group
    dsbayes-group control.csv treated.csv --rope-pct 0.10

    # one file, split on a column
    dsbayes-group events.csv --group-col arm

    # one file, split on dates
    dsbayes-group events.csv --pre 2026-05-29:2026-07-01 \\
                             --post 2026-07-02:2026-08-04

Every row of a CSV is one trial: a unit column (default ``user_id``) and a
0/1 outcome (``convert``). The date split additionally needs a date column
(``datepart``). That is the same shape
:class:`~dstoolbox.ml_funcs.stat_bayes_group.GroupCounts` takes, so there is
no second data path to keep in step.

The first group named is the control: it anchors the ROPE and the relative
lift.

Exit codes: ``0`` the fit ran, ``1`` bad input or a failed check. The
*verdict* is not an exit code — an inconclusive result is a successful run.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .stat_bayes_group import (
    GroupCounts,
    PrePostWindow,
    fit_group_comparison,
    split_by_window,
)

__all__ = ["main"]


def _window_arg(text: str) -> tuple[str, str]:
    """Parse ``START:END`` into a pair of date strings."""
    start, _, end = text.partition(":")
    if not start or not end:
        raise argparse.ArgumentTypeError(
            f"expected START:END (e.g. 2026-05-29:2026-07-01), got {text!r}."
        )
    return start, end


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsbayes-group",
        description=(
            "Does the conversion rate differ between two groups? "
            "Hierarchical Beta-Binomial on unit-level rates."
        ),
        epilog=(
            "The estimand is an unweighted mean over units, so it moves with "
            "a group's composition and not only with behaviour in it. Keep "
            "the groups comparable."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data", nargs="+", metavar="CSV",
        help="one CSV per group (control first), or a single CSV to split "
             "with --group-col or --pre/--post",
    )

    split = parser.add_argument_group("how to split a single CSV")
    split.add_argument(
        "--group-col", metavar="COL",
        help="column holding the group label; must take exactly two values",
    )
    split.add_argument(
        "--control", metavar="VALUE",
        help="which --group-col value is the control "
             "(default: whichever sorts first)",
    )
    split.add_argument(
        "--pre", type=_window_arg, metavar="START:END",
        help="pre-period, both bounds inclusive",
    )
    split.add_argument(
        "--post", type=_window_arg, metavar="START:END",
        help="post-period, both bounds inclusive",
    )

    columns = parser.add_argument_group("columns")
    columns.add_argument("--user-col", default="user_id")
    columns.add_argument("--metric-col", default="convert")
    columns.add_argument("--date-col", default="datepart")

    model = parser.add_argument_group("model")
    model.add_argument(
        "--rope-pct", type=float, default=None, metavar="FRAC",
        help="equivalence band as a fraction of the control rate, e.g. "
             "0.10 for +/-10%%.",
    )
    model.add_argument(
        "--rope-stat", type=float, default=None, metavar="COEF",
        help="equivalence band as a multiple of the control's standard "
             "error, e.g. 0.1 for the Cohen-like 'smaller than noise' band",
    )
    model.add_argument(
        "--rope-biz", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
        help="explicit equivalence band in the units of the rate difference",
    )
    model.add_argument(
        "--min-users", type=int, default=0, metavar="N",
        help="refuse to fit if either group has fewer units than this",
    )
    model.add_argument(
        "--threshold", type=float, default=0.95, metavar="P",
        help="posterior mass needed to call a verdict",
    )
    model.add_argument("--ci", type=float, default=0.95, metavar="P",
                       help="credible interval width")
    model.add_argument("--prior", default="uniform",
                       help="Beta prior on mu: uniform | jeffreys")
    model.add_argument("--draws", type=int, default=2000)
    model.add_argument("--tune", type=int, default=1000)
    model.add_argument("--chains", type=int, default=4)
    model.add_argument("--seed", type=int, default=0)

    out = parser.add_argument_group("output")
    out.add_argument("--plot", metavar="PATH",
                     help="write the posterior figure here (.png / .svg)")
    out.add_argument("--csv", metavar="PATH",
                     help="write the result as a one-row CSV")
    out.add_argument("--quiet", action="store_true",
                     help="print the verdict only")
    return parser


def _groups_from_files(args) -> tuple[GroupCounts, GroupCounts, None]:
    """One CSV per group, control first. Labels come from the filenames."""
    from pathlib import Path  # noqa: PLC0415

    shared = dict(unit_col=args.user_col, metric_col=args.metric_col)
    return (
        *(
            GroupCounts.from_events(
                pd.read_csv(path), label=Path(path).stem, **shared
            )
            for path in args.data
        ),
        None,
    )


def _groups_from_column(args, events: pd.DataFrame) -> tuple[GroupCounts, GroupCounts, None]:
    """Split one CSV on a label column."""
    if args.group_col not in events.columns:
        raise ValueError(f"--group-col {args.group_col!r} is not a column in the data.")

    values = sorted(events[args.group_col].dropna().unique())
    if len(values) != 2:
        raise ValueError(
            f"--group-col {args.group_col!r} takes {len(values)} distinct "
            f"values ({values[:5]}); this compares exactly two."
        )
    if args.control is not None:
        if args.control not in values:
            raise ValueError(f"--control {args.control!r} is not one of {values}.")
        values = [args.control, *[v for v in values if v != args.control]]

    shared = dict(unit_col=args.user_col, metric_col=args.metric_col)
    return (
        *(
            GroupCounts.from_events(
                events, label=str(v), mask=(events[args.group_col] == v).to_numpy(),
                **shared,
            )
            for v in values
        ),
        None,
    )


def _groups_from_dates(args, events: pd.DataFrame):
    """Split one CSV on two date windows."""
    window = PrePostWindow(*args.pre, *args.post)
    pre, post = split_by_window(
        events, window,
        unit_col=args.user_col, metric_col=args.metric_col, date_col=args.date_col,
    )
    return pre, post, window


def _resolve_groups(args):
    """Pick the split the flags asked for, and refuse ambiguous combinations."""
    by_dates = args.pre is not None or args.post is not None
    if len(args.data) > 2:
        raise ValueError(f"expected at most two CSVs, got {len(args.data)}.")
    if len(args.data) == 2:
        if by_dates or args.group_col:
            raise ValueError(
                "two CSVs already define the groups; drop --group-col / --pre / --post."
            )
        return _groups_from_files(args)

    if args.group_col and by_dates:
        raise ValueError("--group-col and --pre/--post are two different splits; pick one.")
    if by_dates and not (args.pre and args.post):
        raise ValueError("--pre and --post go together.")
    if not args.group_col and not by_dates:
        raise ValueError(
            "one CSV needs a split: pass a second CSV, or --group-col, or --pre/--post."
        )

    # Checked the flags first so a bad invocation reports the usage problem
    # rather than whatever the file turns out to be.
    events = pd.read_csv(args.data[0])
    return (
        _groups_from_column(args, events) if args.group_col
        else _groups_from_dates(args, events)
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than raising."""
    args = _build_parser().parse_args(argv)

    try:
        control, treatment, window = _resolve_groups(args)
        effect = fit_group_comparison(
            control,
            treatment,
            metric=args.metric_col,
            window=window,
            prior=args.prior,
            rope_pct_coef=args.rope_pct,
            rope_stat_coef=args.rope_stat,
            rope_biz=None if args.rope_biz is None else tuple(args.rope_biz),
            min_users=args.min_users,
            credibility_threshold=args.threshold,
            hdi_prob=args.ci,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            random_seed=args.seed,
            progressbar=not args.quiet,
        )
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"dsbayes-group: {exc}", file=sys.stderr)
        return 1

    if effect is None:
        print(
            f"dsbayes-group: a group has fewer than --min-users "
            f"{args.min_users} units; no fit was run.",
            file=sys.stderr,
        )
        return 1

    if args.quiet:
        print(effect.decision)
    else:
        print(effect)
        if not effect.converged:
            print(
                f"\nWARNING: sampler did not converge "
                f"(R-hat {effect.rhat_max:.4f}, ESS {effect.ess_min:.0f}, "
                f"{effect.divergences} divergences). Treat the numbers above "
                f"as provisional and re-run with more --draws.",
                file=sys.stderr,
            )

    if args.csv:
        pd.DataFrame([effect.to_row()]).to_csv(args.csv, index=False)
        if not args.quiet:
            print(f"wrote {args.csv}")

    if args.plot:
        from .stat_bayes_group_plots import plot_effect  # noqa: PLC0415

        fig = plot_effect(effect)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        if not args.quiet:
            print(f"wrote {args.plot}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
