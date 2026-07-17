"""Spark spatial grouping: cluster consecutive rows whose successive locations exceed a distance threshold."""

from typing import Callable, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def group_by_proximity(
    df: DataFrame,
    distance_func: Callable,
    x_col: str,
    y_col: str,
    group_by_cols: List[str],
    order_cols: List[str],
    distance_threshold: float = 20.0,
    distance_col_name: str = "distance",
    ordered_ids: bool = False,
) -> DataFrame:
    """Assign a group id to consecutive rows whose location stays within ``distance_threshold``.

    Within each partition defined by ``group_by_cols`` and ordered by ``order_cols``, computes
    the distance between each row's ``(x_col, y_col)`` and the previous row's location using
    ``distance_func``. Rows whose distance is at or above ``distance_threshold`` start a new
    group; otherwise the row joins the previous group.

    Parameters:
    df (DataFrame): Input Spark DataFrame.
    distance_func (Callable): Callable with the signature
        ``(df, point1_coords, point2_coords, distance_col_name=...) -> DataFrame``
        that appends a distance column. See :func:`dstoolbox.spark_funcs.geo.calculate_distance`
        and :func:`dstoolbox.spark_funcs.geo.calculate_haversine_distance`.
    x_col (str): First coordinate column (e.g. X, longitude, easting).
    y_col (str): Second coordinate column (e.g. Y, latitude, northing).
    group_by_cols (List[str]): Partition columns (e.g. entity / device / user IDs).
    order_cols (List[str]): Ordering columns used both for the ``orderBy`` and the window spec.
    distance_threshold (float): Distance in the units returned by ``distance_func`` at or above
        which a new group is started.
    distance_col_name (str): Name for the appended distance column.
    ordered_ids (bool): When True, ``Global_Group_ID`` is a sequential dense rank; otherwise
        it is a fast ``xxhash64`` (recommended for large data).

    Returns:
    DataFrame: The input columns plus ``{distance_col_name}``, ``group_change``,
        ``Local_Group_ID``, ``Global_Group_ID`` and ``Group_Row_Count``.

    """
    df_sorted = df.orderBy(*order_cols)
    window_ordered = Window.partitionBy(*group_by_cols).orderBy(*order_cols)

    prev_x = f"prev_{x_col}"
    prev_y = f"prev_{y_col}"

    df_with_prev = df_sorted.withColumn(prev_x, F.lag(x_col).over(window_ordered)).withColumn(
        prev_y, F.lag(y_col).over(window_ordered)
    )

    df_with_distance = distance_func(
        df_with_prev,
        (x_col, y_col),
        (prev_x, prev_y),
        distance_col_name=distance_col_name,
    )

    df_with_groups = df_with_distance.withColumn(
        "group_change",
        F.when(
            (F.col(distance_col_name) >= F.lit(distance_threshold)) | F.col(prev_x).isNull(),
            1,
        ).otherwise(0),
    ).withColumn(
        "Local_Group_ID",
        F.sum("group_change").over(
            window_ordered.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        ),
    )

    if ordered_ids:
        window_for_global_id = Window.orderBy(*group_by_cols, "Local_Group_ID")
        df_with_groups = df_with_groups.withColumn(
            "Global_Group_ID", F.dense_rank().over(window_for_global_id)
        )
    else:
        df_with_groups = df_with_groups.withColumn(
            "Global_Group_ID",
            F.xxhash64(F.concat_ws("||", *group_by_cols, F.col("Local_Group_ID"))),
        )

    window_group = Window.partitionBy("Global_Group_ID")
    return (
        df_with_groups.withColumn("Group_Row_Count", F.count("*").over(window_group))
        .orderBy(*order_cols, "Local_Group_ID")
    )
