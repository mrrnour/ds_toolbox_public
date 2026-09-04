"""Spark binning: pandas.cut-style discretization of continuous columns."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def cut(
    df: DataFrame,
    input_col: str,
    splits: list[float],
    labels: dict[int, str] | None = None,
    output_col: str | None = None,
    handle_invalid: str = "keep",
) -> tuple[DataFrame, DataFrame]:
    """Bin a continuous Spark column into discrete intervals (pandas.cut equivalent).

    Parameters:
    df (DataFrame): Input Spark DataFrame.
    input_col (str): Name of the numeric column to bin.
    splits (List[float]): Bin boundaries, e.g. ``[-float("inf"), 0, 60, 120, float("inf")]``.
    labels (Optional[Dict[int, str]]): Optional mapping from bin index to label. When set,
        a ``{output_col}_label`` column is added; otherwise only ``{output_col}_index`` is added.
    output_col (Optional[str]): Base name for the added columns. Defaults to ``f"{input_col}_bin"``.
    handle_invalid (str): ``pyspark.ml.feature.Bucketizer`` behaviour for out-of-range or null
        values: ``"keep"``, ``"skip"``, or ``"error"``.

    Returns:
    Tuple[DataFrame, DataFrame]: The binned DataFrame and a frequency table with columns
        ``{output_col}_label``, ``Freq``, ``Percentage`` sorted by descending frequency.

    """
    from pyspark.ml.feature import Bucketizer

    if output_col is None:
        output_col = f"{input_col}_bin"

    bucketizer = Bucketizer(
        splits=splits,
        inputCol=input_col,
        outputCol=f"{output_col}_index",
        handleInvalid=handle_invalid,
    )
    result_df = bucketizer.transform(df)

    if labels is not None:
        label_expr = None
        for bin_idx, label in sorted(labels.items()):
            cond = F.col(f"{output_col}_index") == bin_idx
            label_expr = F.when(cond, label) if label_expr is None else label_expr.when(cond, label)
        label_expr = label_expr.otherwise(None)
        result_df = result_df.withColumn(f"{output_col}_label", label_expr)

    freq_col = f"{output_col}_label" if labels is not None else f"{output_col}_index"
    output_col_freq = (
        result_df.groupBy(freq_col)
        .agg(F.count("*").alias("Freq"))
        .withColumn(
            "Percentage",
            F.round((F.col("Freq") / F.sum("Freq").over(Window.partitionBy())) * 100, 3),
        )
        .orderBy(F.col("Freq").desc())
    )
    return result_df, output_col_freq
