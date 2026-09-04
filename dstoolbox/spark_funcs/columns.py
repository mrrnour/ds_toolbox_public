"""Spark column rename / type-cast utilities."""

import pyspark.sql.functions as F


def rename_cols(df, cols_map):
    """Rename columns of a Spark DataFrame using a mapping dictionary.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Frame to rename.
    cols_map : dict of {str: str}
        Mapping of old column names to new names.

    Returns
    -------
    pyspark.sql.DataFrame
        Frame with columns renamed.
    """
    for old_name, new_name in cols_map.items():
        df = df.withColumnRenamed(old_name, new_name)
    return df


def sp_to_numeric(df, exclude_cols, cast_to="float"):
    """Cast every column of a Spark DataFrame to a numeric type, except the listed columns.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Frame to convert.
    exclude_cols : list of str
        Columns to leave unchanged (typically string keys).
    cast_to : str, optional
        Target Spark numeric type (e.g. ``'float'``, ``'double'``,
        ``'int'``). Default ``'float'``.

    Returns
    -------
    pyspark.sql.DataFrame
        Frame with non-excluded columns cast to ``cast_to``.
    """
    non_str_cols = [col for col in df.columns if col not in exclude_cols]
    for ucol in non_str_cols:
        df = df.withColumn(ucol, F.col(ucol).cast(cast_to))
    return df
