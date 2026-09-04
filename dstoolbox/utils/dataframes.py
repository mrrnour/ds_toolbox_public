"""DataFrame-level utilities: column ops, merges, memory reduction, comparison, encoding bridges, joins."""

import numpy as np
import pandas as pd

from .lists import regex_filter_list
from .text import normalize_text


def movecol(df, cols_to_move=None, ref_col="", place="After"):
    """Reorder ``df`` by moving ``cols_to_move`` next to ``ref_col``.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame.
    cols_to_move : list of str, optional
        Columns to relocate. Defaults to ``[]`` (no-op).
    ref_col : str
        Existing column that ``cols_to_move`` will be placed adjacent to.
    place : {'After', 'Before'}
        Position of ``cols_to_move`` relative to ``ref_col``.

    Returns
    -------
    pandas.DataFrame
        A view of ``df`` with columns re-ordered; original order of
        untouched columns is preserved.
    """
    if cols_to_move is None:
        cols_to_move = []
    cols = df.columns.tolist()
    if place == "After":
        seg1 = cols[: list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == "Before":
        seg1 = cols[: list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]

    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]

    return df[seg1 + seg2 + seg3]


def merge_between(df1, df2, groupCol, closed="both"):
    """Tag each row in ``df1`` with the index of the matching ``[Start, End]`` interval in ``df2``.

    For each group (per ``groupCol``), builds an ``IntervalIndex`` from
    ``df2['Start']`` and ``df2['End']`` and assigns the index of the
    enclosing interval to ``df1['Date']``. Rows whose ``Date`` is not
    inside any interval get index ``-1``.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Must contain ``groupCol`` and a ``Date`` column.
    df2 : pandas.DataFrame
        Must contain ``groupCol``, ``Start``, and ``End`` columns.
    groupCol : str
        Column name used to align rows across the two frames.
    closed : {'left', 'right', 'both', 'neither'}, optional
        Interval-closure semantics. Default ``'both'``.

    Returns
    -------
    pandas.DataFrame
        ``df1`` with an extra ``Index_no`` column.
    """
    #   df1=df_pi_dic_wide
    #   df2=df_cases_edited
    #   groupCol='Vessel'

    df_out = pd.DataFrame(columns=df1.columns.tolist() + ["Index_no"])
    for name, group_df in df1.groupby([groupCol]):
        df2_sub = df2.loc[df2[groupCol] == name]

        #     https://stackoverflow.com/questions/68792511/efficient-way-to-merge-large-pandas-dataframes-between-two-dates
        #     https://stackoverflow.com/questions/31328014/merging-dataframes-based-on-date-range
        #     https://stackoverflow.com/questions/69824730/check-if-value-in-pandas-dataframe-is-within-any-two-values-of-two-other-columns
        #     https://stackoverflow.com/questions/43593554/merging-two-dataframes-based-on-a-date-between-two-other-dates-without-a-common
        #     https://pandas.pydata.org/docs/reference/api/pandas.IntervalIndex.from_arrays.html
        i = pd.IntervalIndex.from_arrays(df2_sub["Start"], df2_sub["End"], closed=closed)
        group_df["Index_no"] = i.get_indexer(group_df["Date"])

        df_out = pd.concat([group_df, df_out], axis=0)

    return df_out


def cell_share_of_total(df, axis=0):
    """Normalize each cell as a fraction of its column (axis=0) or row (axis=1) sum.

    Parameters
    ----------
    df : pandas.DataFrame
        Numeric frame.
    axis : {0, 1}, optional
        ``0`` (default) → divide by column totals; ``1`` → divide by row
        totals.

    Returns
    -------
    pandas.DataFrame
        Same shape as ``df`` with values rescaled to sum to 1 along the
        requested axis.
    """
    if axis == 0:
        out = df.div(df.sum(axis=0), axis=1)
    else:
        out = df.div(df.sum(axis=1), axis=0)
    return out


# ---------------------------------------------------------------------------
# DataFrame column comparison
# ---------------------------------------------------------------------------

_COMPARISON_COL_ORDER = [
    "Column",
    "In_DF1",
    "In_DF2",
    "Type_Match",
    "Value_Commonality_Pct",
    "DF1_Type",
    "DF1_Memory_MB",
    "DF1_Missing_Count",
    "DF1_Missing_Pct",
    "DF2_Type",
    "DF2_Memory_MB",
    "DF2_Missing_Count",
    "DF2_Missing_Pct",
]

_DISPLAY_COL_LABELS = [
    "Column",
    "In DF1",
    "In DF2",
    "Type Match",
    "Value Match %",
    "DF1 Type",
    "DF1 Memory (MB)",
    "DF1 Missing Count",
    "DF1 Missing %",
    "DF2 Type",
    "DF2 Memory (MB)",
    "DF2 Missing Count",
    "DF2 Missing %",
]


def _column_stats(series: pd.Series, prefix: str) -> dict:
    """Per-column dtype/memory/missing stats. Prefix is ``'DF1'`` or ``'DF2'``."""
    missing = int(series.isnull().sum())
    return {
        f"{prefix}_Type": str(series.dtype),
        f"{prefix}_Memory_MB": series.memory_usage(deep=True) / 1024**2,
        f"{prefix}_Missing_Count": missing,
        f"{prefix}_Missing_Pct": (missing / len(series)) * 100 if len(series) else 0.0,
    }


def _empty_column_stats(prefix: str) -> dict:
    """Placeholder stats for a column absent from one side of the comparison."""
    return {
        f"{prefix}_Type": "N/A",
        f"{prefix}_Memory_MB": 0,
        f"{prefix}_Missing_Count": "N/A",
        f"{prefix}_Missing_Pct": "N/A",
    }


def _value_commonality_pct(s1: pd.Series, s2: pd.Series) -> float:
    """Percentage of positionally-matching values between two series.

    Aligned on ``min(len(s1), len(s2))`` and compared elementwise. NaN/NaN
    pairs count as matches. Falls back to 0.0 if dtypes cannot be compared.
    """
    min_length = min(len(s1), len(s2))
    if min_length == 0:
        return 0.0
    s1 = s1.iloc[:min_length].reset_index(drop=True)
    s2 = s2.iloc[:min_length].reset_index(drop=True)
    try:
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            s1_num = pd.to_numeric(s1, errors="coerce")
            s2_num = pd.to_numeric(s2, errors="coerce")
            both_nan = s1_num.isna() & s2_num.isna()
            both_valid = ~s1_num.isna() & ~s2_num.isna()
            matches = both_nan | (np.isclose(s1_num, s2_num, equal_nan=False) & both_valid)
        else:
            s1_str, s2_str = s1.astype(str), s2.astype(str)
            matches = (s1_str == s2_str) | ((s1_str == "nan") & (s2_str == "nan"))
        return round((matches.sum() / min_length) * 100, 1)
    except (TypeError, ValueError):
        return 0.0


def _comparison_row(col: str, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    """Build one row of the column-comparison DataFrame."""
    in_df1, in_df2 = col in df1.columns, col in df2.columns

    if in_df1 and in_df2:
        type_match = "✓" if df1[col].dtype == df2[col].dtype else "✗"
        commonality = _value_commonality_pct(df1[col], df2[col])
    else:
        type_match = "N/A"
        commonality = "N/A"

    row = {
        "Column": col,
        "In_DF1": "✓" if in_df1 else "✗",
        "In_DF2": "✓" if in_df2 else "✗",
        "Type_Match": type_match,
        "Value_Commonality_Pct": commonality,
    }
    row.update(_column_stats(df1[col], "DF1") if in_df1 else _empty_column_stats("DF1"))
    row.update(_column_stats(df2[col], "DF2") if in_df2 else _empty_column_stats("DF2"))
    return row


def _sort_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by descending commonality, then column name (case-insensitive)."""

    def key(row):
        commonality = (
            -1 if row["Value_Commonality_Pct"] == "N/A" else float(row["Value_Commonality_Pct"])
        )
        return (-commonality, str(row["Column"]).lower())

    order = sorted(range(len(df)), key=lambda i: key(df.iloc[i]))
    return df.iloc[order].reset_index(drop=True)


def _format_display_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Rename + format numbers for pretty console display."""
    out = comparison_df.copy()
    out.columns = _DISPLAY_COL_LABELS
    for col in ("DF1 Memory (MB)", "DF2 Memory (MB)"):
        out[col] = out[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and x != 0 else str(x)
        )
    for col in ("DF1 Missing %", "DF2 Missing %", "Value Match %"):
        out[col] = out[col].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x))
    return out


def _dtype_summary(
    df1: pd.DataFrame, df2: pd.DataFrame, df1_name: str, df2_name: str
) -> pd.DataFrame:
    """Count of columns by dtype for each frame."""
    df1_types = df1.dtypes.value_counts().to_dict()
    df2_types = df2.dtypes.value_counts().to_dict()
    all_types = sorted(set(df1_types) | set(df2_types), key=str)
    return pd.DataFrame(
        [
            {
                "Data_Type": str(dt),
                f"{df1_name}_Count": df1_types.get(dt, 0),
                f"{df2_name}_Count": df2_types.get(dt, 0),
            }
            for dt in all_types
        ]
    )


def _print_comparison_report(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str,
    df2_name: str,
    comparison_df: pd.DataFrame,
    common_columns: set,
    df1_only: list[str],
    df2_only: list[str],
    type_matches: int,
    type_match_pct: float,
) -> None:
    """Console display for the full comparison report."""
    print(f"{'='*80}\nDATAFRAME COLUMN COMPARISON: {df1_name} vs {df2_name}\n{'='*80}")
    print(f"\n📊 BASIC INFORMATION\n{'-'*50}")
    print(f"{df1_name}: {df1.shape[0]:,} rows × {df1.shape[1]:,} columns")
    print(f"{df2_name}: {df2.shape[0]:,} rows × {df2.shape[1]:,} columns")

    display_df = _format_display_table(comparison_df)
    print(f"\n📋 DETAILED COLUMN COMPARISON\n{display_df.to_string(index=False)}")

    print("\n📈 COLUMN SUMMARY")
    print(f"Total unique columns: {len(comparison_df)}")
    print(f"Common columns: {len(common_columns)}")
    print(f"Only in {df1_name}: {len(df1_only)}")
    print(f"Only in {df2_name}: {len(df2_only)}")

    if common_columns:
        print(
            f"Common columns with matching types: {type_matches}/{len(common_columns)} ({type_match_pct:.1f}%)"
        )
        commonalities = [float(v) for v in comparison_df["Value_Commonality_Pct"] if v != "N/A"]
        if commonalities:
            print(f"Average value commonality: {sum(commonalities)/len(commonalities):.1f}%")

    if df1_only:
        print(f"\nColumns only in {df1_name}: {sorted(df1_only)}")
    if df2_only:
        print(f"Columns only in {df2_name}: {sorted(df2_only)}")

    mem1 = comparison_df.loc[comparison_df["DF1_Memory_MB"] != 0, "DF1_Memory_MB"].sum()
    mem2 = comparison_df.loc[comparison_df["DF2_Memory_MB"] != 0, "DF2_Memory_MB"].sum()
    print(f"\n🎯 MEMORY USAGE SUMMARY\n{'='*50}")
    print(f"{df1_name} total memory: {mem1:.3f} MB")
    print(f"{df2_name} total memory: {mem2:.3f} MB")
    if mem1 > 0 and mem2 > 0:
        diff_pct = ((mem2 - mem1) / mem1) * 100
        print(f"Memory difference: {diff_pct:+.1f}% ({mem2 - mem1:+.3f} MB)")


def compare_dataframes_columns(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str = "DataFrame 1",
    df2_name: str = "DataFrame 2",
    display: bool = True,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Compare columns between two DataFrames.

    Reports per-column presence, dtype match, value commonality, memory,
    and missingness. Optionally prints a formatted report.

    Returns
    -------
    tuple
        ``(comparison_table, summary_dict, type_summary)``.
    """
    all_columns = sorted(set(df1.columns) | set(df2.columns))
    common_columns = set(df1.columns) & set(df2.columns)
    df1_only = sorted(set(df1.columns) - set(df2.columns))
    df2_only = sorted(set(df2.columns) - set(df1.columns))

    rows = [_comparison_row(col, df1, df2) for col in all_columns]
    comparison_df = _sort_comparison_table(pd.DataFrame(rows, columns=_COMPARISON_COL_ORDER))

    type_matches = sum(1 for col in common_columns if df1[col].dtype == df2[col].dtype)
    type_match_pct = (type_matches / len(common_columns) * 100) if common_columns else 0.0

    if display:
        _print_comparison_report(
            df1,
            df2,
            df1_name,
            df2_name,
            comparison_df,
            common_columns,
            df1_only,
            df2_only,
            type_matches,
            type_match_pct,
        )

    summary_dict = {
        "basic_info": {
            f"{df1_name}_shape": df1.shape,
            f"{df2_name}_shape": df2.shape,
        },
        "common_columns": list(common_columns),
        "df1_only_columns": df1_only,
        "df2_only_columns": df2_only,
        "type_matches": type_matches,
        "type_match_percentage": type_match_pct,
        "total_memory_df1_mb": comparison_df.loc[
            comparison_df["DF1_Memory_MB"] != 0, "DF1_Memory_MB"
        ].sum(),
        "total_memory_df2_mb": comparison_df.loc[
            comparison_df["DF2_Memory_MB"] != 0, "DF2_Memory_MB"
        ].sum(),
    }
    return comparison_df, summary_dict, _dtype_summary(df1, df2, df1_name, df2_name)


def categorical_to_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all object/category columns of ``df`` to integer category codes.

    Returns a new DataFrame; the input is not mutated.
    """
    out = df.copy()
    cat_columns = (
        out.select_dtypes(include=["object"]).columns.tolist()
        + out.select_dtypes(include=["category"]).columns.tolist()
    )
    if cat_columns:
        out[cat_columns] = out[cat_columns].apply(lambda x: x.astype("category").cat.codes)
    return out


_PLAN_COLS = ("column", "original_dtype", "target_dtype", "reason")


def _select_int_dtype(c_min, c_max) -> str | None:
    """Smallest int dtype fitting ``[c_min, c_max]`` or None if int64 already fits best."""
    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
        return "int8"
    if c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
        return "int16"
    if c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
        return "int32"
    return None


def _select_float_dtype(c_min, c_max, use_float16: bool) -> str | None:
    """Smallest float dtype fitting ``[c_min, c_max]``; ``None`` if float64 needed."""
    if use_float16 and abs(c_max - c_min) < 65504:
        f16 = np.finfo(np.float16)
        if c_min >= f16.min and c_max <= f16.max:
            return "float16"
    f32 = np.finfo(np.float32)
    if c_min >= f32.min and c_max <= f32.max:
        return "float32"
    return None


def _plan_column(
    series: pd.Series,
    column: str,
    obj2str_cols: list[str] | str,
    str2cat_cols: list[str] | str,
    use_float16: bool,
) -> dict:
    """Return one plan row: ``{column, original_dtype, target_dtype, reason}``.

    ``target_dtype`` is ``None`` when the column should be left alone.
    """
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    original_dtype = str(series.dtype)

    def row(target, reason):
        return {
            "column": column,
            "original_dtype": original_dtype,
            "target_dtype": target,
            "reason": reason,
        }

    if is_datetime(series):
        return row(None, "datetime column — left untouched")

    if pd.api.types.is_integer_dtype(series):
        if series.empty or series.isna().all():
            return row(None, "no integer values to fit")
        target = _select_int_dtype(series.min(), series.max())
        if target is None:
            return row(None, f"int range exceeds int32; keep {original_dtype}")
        return row(target, f"int range fits {target}")

    if pd.api.types.is_float_dtype(series):
        if not np.isfinite(series).all():
            return row(None, "float column contains inf/NaN — left untouched")
        target = _select_float_dtype(series.min(), series.max(), use_float16)
        if target is None:
            return row(None, f"float range exceeds float32; keep {original_dtype}")
        return row(target, f"float range fits {target}")

    selects_obj = (obj2str_cols == "all_columns") or (column in obj2str_cols)
    selects_cat = (str2cat_cols == "all_columns") or (column in str2cat_cols)

    if series.dtype == object:
        if selects_obj and selects_cat:
            return row("category", "object → string → category")
        if selects_obj:
            return row("string", "object → string")
        return row(None, "object column not selected for conversion")

    if pd.api.types.is_string_dtype(series):
        if selects_cat:
            return row("category", "string → category")
        return row(None, "string column not selected for conversion")

    return row(None, f"dtype {original_dtype} has no reduction rule")


def _plan_memory_reduction(
    df: pd.DataFrame,
    obj2str_cols: list[str] | str = "all_columns",
    str2cat_cols: list[str] | str = "all_columns",
    use_float16: bool = False,
) -> pd.DataFrame:
    """Decide what dtype each column should become to shrink memory. Pure.

    Returns one row per column of ``df`` with the fields
    ``[column, original_dtype, target_dtype, reason]``. ``target_dtype`` is
    ``None`` for columns that should be left alone. No mutation, no I/O.
    Apply with :func:`_apply_memory_reduction`, or use
    :func:`reduce_mem_usage` for plan + apply + report in one call.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame to analyze.
    obj2str_cols : list of str or ``"all_columns"``
        Columns to consider for object → string conversion.
    str2cat_cols : list of str or ``"all_columns"``
        Columns to consider for string → category conversion.
    use_float16 : bool
        If True, allow float16 (precision-lossy). Float32 otherwise.
    """
    rows = [
        _plan_column(df[col], col, obj2str_cols, str2cat_cols, use_float16) for col in df.columns
    ]
    return pd.DataFrame(rows, columns=list(_PLAN_COLS))


def _apply_memory_reduction(df: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Apply a plan DataFrame to ``df``; returns a new frame.

    Rows where ``plan['target_dtype']`` is null are copied through
    unchanged. Does not mutate the input.
    """
    out = df.copy()
    changes = plan.loc[plan["target_dtype"].notna(), ["column", "target_dtype"]]
    for column, target_dtype in changes.itertuples(index=False):
        out[column] = out[column].astype(target_dtype)
    return out


def reduce_mem_usage(
    df,
    obj2str_cols="all_columns",
    str2cat_cols="all_columns",
    use_float16: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Convenience: plan, apply, and print a before/after memory summary.

    Returns a new optimized DataFrame; the input is not mutated.
    """
    plan = _plan_memory_reduction(df, obj2str_cols, str2cat_cols, use_float16)
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")
    if verbose:
        for _, r in plan[plan["target_dtype"].notna()].iterrows():
            print(f"{r['column']}: {r['reason']}")
    out = _apply_memory_reduction(df, plan)
    end_mem = out.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    if start_mem:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"Decreased by {reduction:.1f}%")
    return out


def null_per_column(df):
    """Compute the percentage of nulls in each column, sorted descending.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame.

    Returns
    -------
    pandas.DataFrame
        Single-column frame named ``null_percent``, indexed by column
        name and sorted high → low.
    """
    null_per = df.isnull().sum() / df.shape[0] * 100
    null_per = pd.DataFrame(null_per, columns=["null_percent"])
    null_per = (
        null_per.reset_index()
        .sort_values(by=["null_percent", "index"], ascending=False)
        .set_index("index")
    )
    return null_per


def unify_cols(df1, df2, df1_name, df2_name):
    """Align two DataFrames so they share the same column set, padding with zeros.

    Any column in ``df1`` but not ``df2`` is added to ``df2`` filled
    with zeros, and vice-versa. Both frames end up with the same column
    order. ``df1.index`` is reset to ``df2.index`` for alignment.

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        Frames to unify.
    df1_name, df2_name : str
        Display names used in the printed "Adding following columns to
        ..." message.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(df1, df2)`` with identical columns.
    """
    df1.index = df2.index

    def unify_cols__sub(df1, df2, df1_name, df2_name):
        """Add any of ``df1``'s missing columns to ``df2`` as zero-filled columns."""
        diff1 = np.setdiff1d(df1.columns, df2.columns)
        if diff1.size != 0:
            print(f"Adding following columns to {df2_name} as there are in {df1_name}:\n {diff1}")
            df2 = pd.concat([df2, pd.DataFrame(0, index=df2.index, columns=diff1)], axis=1)
            df2 = df2[df1.columns]
        return df2

    df2 = unify_cols__sub(df1, df2, df1_name, df2_name)
    df1 = unify_cols__sub(df2, df1, df2_name, df1_name)
    return df1, df2


def percent_agg(df, grpby1, grpby2, sumCol):
    """Compute each ``grpby1`` total as a percentage of its parent ``grpby2`` total.

    Useful for sub-aggregate share calculations (e.g. share of revenue
    per product within each region).

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame.
    grpby1 : list of str
        Fine-grained group key (must include ``grpby2`` as a subset).
    grpby2 : list of str
        Coarse group key used as the percentage denominator.
    sumCol : str
        Column to sum.

    Returns
    -------
    pandas.DataFrame
        Long-format frame with ``grpby1`` columns plus ``<sumCol>_percent``
        and ``<sumCol>``; rows where the percentage is zero are dropped.
    """
    agg1 = df.groupby(grpby1)[sumCol].sum().reset_index()
    agg2 = df.groupby(grpby2)[sumCol].sum().reset_index()

    agg1 = df.groupby(grpby1)[sumCol].sum()
    agg1 = agg1.groupby(level=grpby2).apply(lambda x: 100 * x / float(x.sum())).reset_index()
    agg1.rename(columns={sumCol: f"{sumCol}_percent"}, inplace=True)

    agg1 = agg1[agg1[f"{sumCol}_percent"] != 0]
    #   agg1=agg1.merge(agg2,on=grpby2)
    #   agg1[f'{sumCol}_percent']=np.round(agg1[f'{sumCol}_x']/agg1[f'{sumCol}_y']*100,0)
    #   agg1=agg1[agg1[outCol]!=0]

    ##NOTE:
    #   agg1.div(agg2, level=grpby2) * 100  doesnot work

    ##print(agg1.groupby(grpby2)[f"{sumCol}_percent"].sum())
    agg1[f"{sumCol}"] = pd.Series(df.groupby(grpby1)[sumCol].sum().values)

    return agg1


def fill_with_colnames(udata):
    """Replace every non-zero cell with its column name; zero cells stay ``0``.

    Useful as the first step for building a comma-joined feature summary
    per row (see :func:`condense_cols`).

    Parameters
    ----------
    udata : pandas.DataFrame
        Numeric frame (``samples × features``). Values are compared
        against 0 via ``astype(int)``.

    Returns
    -------
    pandas.DataFrame
        Same shape and index as ``udata``. Each cell is either the
        column name (str) if the original value was non-zero, or the
        integer ``0``.
    """

    tmp = np.tile(udata.columns, [len(udata.index), 1])
    tmp2 = pd.DataFrame(
        np.where(udata.astype(int), tmp, 0), columns=udata.columns, index=udata.index
    )
    # tmp2 = tmp2.replace(0, "")
    return tmp2


def join_non_zero(x, sep=", "):
    """Join items of ``x`` with ``sep``, skipping any item equal to ``0``.

    Designed to consume rows produced by :func:`fill_with_colnames`,
    whose cells are either the column name (str) or the integer ``0``.
    The filter uses ``item != 0`` so string cells (which are never ``==
    0``) survive and get joined.

    Parameters
    ----------
    x : iterable
        Row of mixed strings and zeros.
    sep : str, optional
        Separator inserted between kept items. Default ``', '``.

    Returns
    -------
    str
        Concatenation of the non-zero items.
    """
    y = sep.join(list(filter(lambda xx: xx != 0, x)))
    return y


def clean_product_descriptions(prodDesc, df):
    """Deduplicate ``prodDesc`` and align it to the product columns of ``df``.

    Extracts ``['PRODUCT_ID', 'LOB']`` from ``prodDesc``, drops duplicate
    rows, uppercases both columns, then left-joins onto the uppercased
    column names of ``df`` (treated as ``PRODUCT_ID``s). The resulting
    frame has exactly one row per column of ``df``, in column order.

    Parameters
    ----------
    prodDesc : pandas.DataFrame
        Product catalogue; must contain ``PRODUCT_ID`` and ``LOB``
        (line-of-business) columns.
    df : pandas.DataFrame
        Feature frame whose columns are ``PRODUCT_ID`` values.

    Returns
    -------
    pandas.DataFrame
        Rows = ``df.columns``; columns = ``['PRODUCT_ID', 'LOB']``.
        Rows for products not found in ``prodDesc`` have NaN ``LOB``.

    Raises
    ------
    ValueError
        If the joined frame's row count does not equal ``df.shape[1]``,
        indicating a merge inconsistency.
    """

    prodDesc2 = prodDesc[["PRODUCT_ID", "LOB"]].drop_duplicates()
    prodDesc2["PRODUCT_ID"] = prodDesc2["PRODUCT_ID"].str.upper()
    prodDesc2["LOB"] = prodDesc2["LOB"].str.upper()

    tmp = pd.DataFrame(df.columns.str.upper(), columns=["PRODUCT_ID"])
    prodDesc2 = tmp.merge(prodDesc2, on="PRODUCT_ID", how="left")

    if prodDesc2.shape[0] != df.shape[1]:
        print(prodDesc2)
        print(df.columns)
        raise ValueError(
            "prodDesc2 row count does not match df column count; " "check PRODUCT_ID coverage."
        )

    return prodDesc2


def condense_cols(df, remove_prefix, umap):
    """Collapse each row of a wide 0/non-zero frame into a comma-joined string of active column names.

    Pipeline: optionally strip the first ``<prefix>_`` segment from each
    column name, rename via ``umap``, replace non-zero cells with their
    (renamed) column name via :func:`fill_with_colnames`, then join the
    surviving names row-wise with ``", "``.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide ``samples × features`` numeric frame.
    remove_prefix : bool
        If True, drop the leading ``<prefix>_`` segment from every column
        name before joining.
    umap : dict
        Column-name remapping applied after ``remove_prefix``; pass
        ``{}`` to skip.

    Returns
    -------
    pandas.Series
        One row per input row, each a comma-joined string of the column
        names whose value was non-zero in that row.
    """

    if remove_prefix:
        df.columns = pd.Series(["_".join(x.split("_")[1:]) for x in df.columns])
    if len(umap) != 0:
        df.rename(columns=umap, inplace=True)
    df2 = fill_with_colnames(df)
    df2 = df2.apply(join_non_zero, sep=", ", axis=1)
    return df2


def _rank_products(x, y):
    """Rank product classes by descending predicted probability (single sample).

    Internal helper for multi-label product recommendation output. Given
    one row of ``classifier.predict_proba`` and the matching class
    labels, sort both in descending order of probability and return them
    concatenated in one flat list; class names for zero-probability
    entries are blanked to ``''`` so the row is publish-ready as a wide
    spreadsheet record.

    Parameters
    ----------
    x : pandas.Series
        One row of ``classifier.predict_proba`` — length ``n_classes``,
        values in ``[0, 1]``.
    y : pandas.Series
        Class labels aligned with ``x`` (typically ``clf.classes_`` or
        the columns of the ``predict_proba`` DataFrame).

    Returns
    -------
    list
        Flat list of length ``2 * n_classes``:
        ``[class_1, class_2, ..., '', '', prob_1, prob_2, ..., 0.0, 0.0]``
        — first half is class labels sorted by descending probability
        (blanked where prob is 0), second half is the sorted
        probabilities.
    """
    x = x.values
    y = y.values
    idx = np.argsort(-x)
    x = x[idx]
    y = y[idx]

    idx = np.argwhere(x == 0)
    y[idx] = ""

    return y.tolist() + x.tolist()


def encode_categoricals(df, cat_decoder="OneHotEncoder"):
    """Encode object/category columns to numeric using one-hot or category codes.

    Strings are first lower-cased and stripped. With ``cat_decoder='OneHotEncoder'``
    the categorical block is replaced with a one-hot expansion; otherwise
    each column is replaced with ``Categorical.cat.codes``.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame; modified and returned.
    cat_decoder : {'OneHotEncoder', any other}, optional
        Encoder strategy. Default ``'OneHotEncoder'``.

    Returns
    -------
    pandas.DataFrame
        Frame with categorical columns replaced by their numeric encoding.
    """
    # -------Conversion cat to numerical
    # TODO:add to functions or use existing lib
    cat_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_columns) != 0:
        df[cat_columns] = df[cat_columns].applymap(
            lambda x: str(x).lower().strip() if not pd.isnull(x) else x
        )  # lower case
        print("Categorical columns...")
        tmp = df[cat_columns].nunique()
        tmp.sort_values(inplace=True, ascending=False)
        print(tmp)
        ### debugging:
        # df=df0.copy()
        # cat_columns = df.select_dtypes(include=['object']).columns.tolist()+df.select_dtypes(include=['category']).columns.tolist()

        if cat_decoder == "OneHotEncoder":
            df[cat_columns] = df[cat_columns].fillna("None").astype("category")
            from sklearn.preprocessing import OneHotEncoder

            enc = OneHotEncoder()
            tmp = enc.fit_transform(df[cat_columns])

            tmp2 = pd.DataFrame(
                tmp.todense(), columns=enc.get_feature_names(cat_columns), index=df.index
            )
            df = pd.concat([df.drop(cat_columns, axis=1), tmp2], axis=1)
        else:
            df[cat_columns] = df[cat_columns].apply(lambda x: x.astype("category").cat.codes)

    return df


def flexible_join(left_df, right_df, left_on=None, right_on=None, on=None, how="inner", **kwargs):
    """
    Join two DataFrames with flexible string matching that handles differences in:
    - spaces, underscores, and other special characters (/, -, etc.)
    - letter case (upper/lower)

    Parameters:
    -----------
    left_df : pandas DataFrame
        Left DataFrame to join
    right_df : pandas DataFrame
        Right DataFrame to join
    left_on : str or list of str, optional
        Column(s) from left_df to use as join key(s)
    right_on : str or list of str, optional
        Column(s) from right_df to use as join key(s)
    on : str or list of str, optional
        Column name(s) to join on if column names are identical in both DataFrames
    how : str, default 'inner'
        Type of join to perform ('inner', 'left', 'right', 'outer')
    **kwargs :
        Additional keyword arguments to pass to pd.merge()

    Returns:
    --------
    pandas DataFrame
        Joined DataFrame
    """
    # Create copies to avoid modifying the original DataFrames
    left_copy = left_df.copy()
    right_copy = right_df.copy()

    # Handle the case where 'on' is specified
    if on is not None:
        left_on = right_on = on

    # Convert single column to list
    if isinstance(left_on, str):
        left_on = [left_on]
    if isinstance(right_on, str):
        right_on = [right_on]

    # Make sure we have valid join columns
    if left_on is None or right_on is None:
        raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")

    # Make sure the lengths match
    if len(left_on) != len(right_on):
        raise ValueError("Length of 'left_on' must equal length of 'right_on'")

    # Create normalized versions of each join column
    left_norm_cols = []
    right_norm_cols = []

    for lcol, rcol in zip(left_on, right_on, strict=False):
        # Create normalized column names that include the original column names
        left_norm_col = f"_normalized_left_{lcol}"
        right_norm_col = f"_normalized_right_{rcol}"

        # Add to our lists of normalized columns
        left_norm_cols.append(left_norm_col)
        right_norm_cols.append(right_norm_col)

        # Create the normalized columns
        left_copy[left_norm_col] = left_copy[lcol].apply(normalize_text)
        right_copy[right_norm_col] = right_copy[rcol].apply(normalize_text)

    # Perform the join on the normalized keys
    result = pd.merge(
        left_copy, right_copy, left_on=left_norm_cols, right_on=right_norm_cols, how=how, **kwargs
    )

    # Drop the temporary normalized key columns
    result = result.drop(columns=left_norm_cols + right_norm_cols)

    return result


def dates_to_months_since_min(df, dateCols):
    """Replace date columns in-place with integer months since the earliest date.

    Only columns whose name matches one of ``dateCols`` (via
    :func:`regex_filter_list`) are touched. Each matched column is cast to
    ``datetime64[ns]``, then transformed to
    ``(value - global_min) / 1_month`` and rounded to ``int64``. The
    ``global_min`` is the minimum of the *first* matched date column
    (the ``.min()[0]`` in the implementation), so all matched columns
    share the same origin. Missing values are preserved as NaN.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame; date columns are mutated in place.
    dateCols : list of str
        Regex patterns matched against ``df.columns`` to identify date
        columns.

    Returns
    -------
    pandas.DataFrame
        The same ``df`` with matched date columns replaced by integer
        month offsets from the earliest date (NaN preserved).
    """

    tmploc, _ = regex_filter_list(dateCols, df.columns.values)

    if len(tmploc) != 0:
        df[tmploc] = df[tmploc].astype("datetime64[ns]")

        print("conversion date features to number (month): date - " + str(df[tmploc].min()[0]))
        tmp3 = df[tmploc].apply(lambda x: x - df[tmploc].min()[0])
        tmp3 = tmp3.apply(lambda x: x / np.timedelta64(1, "M"))
        df[tmploc] = tmp3.fillna(-1).round(0).astype("int64")
        df[tmploc] = df[tmploc].replace(-1, np.nan)

    return df
