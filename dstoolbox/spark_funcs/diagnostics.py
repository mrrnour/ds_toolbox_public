"""Spark data-quality diagnostics: missing-value profiling, duplicate detection, anti-join lookups."""

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def percent_missing(df: DataFrame) -> DataFrame:
    """Return the fraction of null (and NaN, for float/double columns) entries per column.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input Spark DataFrame.

    Returns
    -------
    pyspark.sql.DataFrame
        One-row DataFrame with one column per input column, each value
        in ``[0.0, 1.0]``.
    """
    exprs = []
    for c, t in df.dtypes:
        if t in ("double", "float"):
            expr = (
                F.count(F.when(F.col(c).isNull() | F.isnan(F.col(c)), c)) / F.count(F.lit(1))
            ).alias(c)
        else:
            expr = (F.count(F.when(F.col(c).isNull(), c)) / F.count(F.lit(1))).alias(c)
        exprs.append(expr)
    return df.select(*exprs)


def find_duplicates(
    df: DataFrame,
    duplicate_cols: list[str],
    add_row_number: bool = False,
    order_by_cols: list[str] | None = None,
) -> DataFrame:
    """Return all rows that share a value on ``duplicate_cols`` with at least one other row.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input Spark DataFrame.
    duplicate_cols : list of str
        Columns used to define the duplicate key.
    add_row_number : bool, default False
        If True, add a ``row_number`` column partitioned by
        ``duplicate_cols`` and ordered by ``order_by_cols`` (or
        ``duplicate_cols`` when unset).
    order_by_cols : list of str, optional
        Ordering used both for the added row-number window and the
        final ``orderBy``. When None, the result is ordered by
        ``duplicate_cols`` + ``duplicate_count`` (or ``row_number``
        when ``add_row_number=True``).

    Returns
    -------
    pyspark.sql.DataFrame
        All duplicated rows, augmented with a ``duplicate_count``
        column and, optionally, a ``row_number`` column.
    """
    dup_counts = (
        df.groupBy(duplicate_cols)
        .agg(F.count("*").alias("duplicate_count"))
        .filter(F.col("duplicate_count") > 1)
    )

    df2 = df.join(dup_counts, on=duplicate_cols, how="left")
    df2 = df2.fillna(1, subset=["duplicate_count"])

    if add_row_number:
        order_cols = order_by_cols if order_by_cols else duplicate_cols
        df2 = df2.withColumn(
            "row_number",
            F.row_number().over(Window.partitionBy(duplicate_cols).orderBy(order_cols)),
        )

    if order_by_cols:
        return df2.orderBy(*order_by_cols)
    if add_row_number:
        return df2.orderBy(*(duplicate_cols + ["row_number"]))
    return df2.orderBy(*(duplicate_cols + ["duplicate_count"]))


def find_lost_records(
    df1: DataFrame,
    df2: DataFrame,
    key_columns: str | list[str],
    limit: int = 10,
    display_first: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, DataFrame | None]:
    """Return keys present in ``df1`` but missing from ``df2`` (left-anti join preview).

    Parameters:
    df1 (DataFrame): Source Spark DataFrame.
    df2 (DataFrame): Target Spark DataFrame to compare against.
    key_columns (Union[str, List[str]]): Column name or list of column names forming the key.
    limit (int): Maximum number of lost key combinations to return.
    display_first (bool): When True, also return the full ``df1`` rows matching the first
        lost key combination.
    verbose (bool): When True, print a short human-readable summary and (if
        ``display_first``) show the first matched records.

    Returns:
    Tuple[pandas.DataFrame, Optional[DataFrame]]: The pandas DataFrame of lost keys and,
        when ``display_first=True`` and lost keys exist, the Spark DataFrame of full rows
        for the first lost key; otherwise ``None`` for the second element.

    """
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    elif not isinstance(key_columns, list):
        raise ValueError("key_columns must be a string or list of strings")

    lost_keys_spark = (
        df1.select(key_columns)
        .distinct()
        .join(df2.select(key_columns).distinct(), on=key_columns, how="left_anti")
        .limit(limit)
    )
    lost_keys_df = lost_keys_spark.toPandas()

    if lost_keys_df.empty:
        if verbose:
            print(
                f"No lost records - all key combinations from df1 exist in df2 (keys: {', '.join(key_columns)})."
            )
        return lost_keys_df, None

    if verbose:
        print(f"Lost records found based on key(s): {', '.join(key_columns)}")
        print(f"Number of lost key combinations (up to limit={limit}): {len(lost_keys_df)}")
        print("First few lost key combinations:")
        print(lost_keys_df.head().to_string(index=False))

    if not display_first:
        return lost_keys_df, None

    first_row = lost_keys_df.iloc[0]
    filter_condition = None
    for col in key_columns:
        value = first_row[col]
        if hasattr(value, "item"):
            value = value.item()
        col_condition = F.col(col) == value
        filter_condition = (
            col_condition if filter_condition is None else filter_condition & col_condition
        )

    result_df = df1.filter(filter_condition)
    if verbose:
        print("Full records for first lost key in df1:")
        result_df.show(truncate=False)
    return lost_keys_df, result_df
