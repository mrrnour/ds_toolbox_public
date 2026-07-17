"""Spark wide↔long reshape helpers."""

from typing import List

import pyspark.sql.functions as F
from pyspark.sql import DataFrame as DataFrame_ps


def melt(df: DataFrame_ps,
        id_vars: List[str], value_vars: List[str],
        var_name: str="variable", value_name: str="value") -> DataFrame_ps:
    """Unpivot a wide Spark DataFrame into long form (PySpark equivalent of ``pandas.melt``).

    Based on the pattern from https://stackoverflow.com/questions/41670103/how-to-melt-spark-dataframe.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Wide-form input.
    id_vars : list of str
        Columns to use as identifier variables (kept as-is in the output).
    value_vars : list of str
        Columns to unpivot into ``(var_name, value_name)`` pairs.
    var_name : str, default ``'variable'``
        Name for the melted variable column.
    value_name : str, default ``'value'``
        Name for the melted value column.

    Returns
    -------
    pyspark.sql.DataFrame
        Long-form frame with columns ``id_vars + [var_name, value_name]``.

    Notes
    -----
    Does not currently accept ``None`` for ``id_vars`` / ``value_vars``
    (unlike ``pandas.melt``).
    """

    # Create an array of structs with all columns to be unpivoted
    # leaving just two non-identifier columns, ‘variable’ and ‘value’.
    _vars_and_vals = F.array(*(
        F.struct(F.lit(c).alias(var_name), F.col(c).alias(value_name))
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", F.explode(_vars_and_vals))

    # Select the required columns
    cols = id_vars + [
        F.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]
    ]

    return _tmp.select(*cols)
