"""Temporal-event helpers on Spark: LEAD/LAG windowing, overlap analysis, event merging."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def prepare_consecutive_events(
    df: DataFrame,
    partition_cols: list[str],
    order_col: str,
    start_time_col: str,
    end_time_col: str,
) -> DataFrame:
    """Add LEAD/LAG columns to a DataFrame so consecutive-event analysis can compare rows.

    Adds ``Next_{start_time_col}``, ``Next_{end_time_col}``, ``Previous_{start_time_col}``,
    ``Previous_{end_time_col}`` computed with a window partitioned by ``partition_cols`` and
    ordered by ``order_col``.

    Parameters:
    df (DataFrame): Input Spark DataFrame.
    partition_cols (List[str]): Columns to partition the window over (e.g. entity / device / user IDs).
    order_col (str): Column to order the window by; typically the start-time column.
    start_time_col (str): Name of the start-timestamp column to LEAD/LAG.
    end_time_col (str): Name of the end-timestamp column to LEAD/LAG.

    Returns:
    DataFrame: Input DataFrame plus the four ``Next_``/``Previous_`` timestamp columns
        whose names are derived from ``start_time_col`` and ``end_time_col``.

    """
    window_spec = Window.partitionBy(*partition_cols).orderBy(F.col(order_col))
    return (
        df.withColumn(f"Next_{start_time_col}", F.lead(F.col(start_time_col), 1).over(window_spec))
        .withColumn(f"Next_{end_time_col}", F.lead(F.col(end_time_col), 1).over(window_spec))
        .withColumn(f"Previous_{start_time_col}", F.lag(F.col(start_time_col), 1).over(window_spec))
        .withColumn(f"Previous_{end_time_col}", F.lag(F.col(end_time_col), 1).over(window_spec))
    )


def analyze_temporal_overlaps(
    df: DataFrame,
    start_time_col: str,
    end_time_col: str,
    next_start_col: str,
    next_end_col: str,
    prev_start_col: str | None = None,
    prev_end_col: str | None = None,
    output_columns: list[str] | None = None,
    time_unit: str = "seconds",
    adjacency_tolerance: int = 0,
) -> DataFrame:
    """Compute overlap / gap / adjacency between two event windows on each row.

    Two modes:

    - **Independent mode** (default): compare event A (``start_time_col``, ``end_time_col``)
      against event B (``next_start_col``, ``next_end_col``). Leave ``prev_*`` as ``None``.
    - **Consecutive mode**: pass all four ``next_*`` / ``prev_*`` columns; comparison uses
      the next event when available and falls back to the previous event otherwise.
      Use :func:`prepare_consecutive_events` to populate the next/previous columns.

    Parameters:
    df (DataFrame): Input Spark DataFrame.
    start_time_col (str): Event-A start timestamp column.
    end_time_col (str): Event-A end timestamp column.
    next_start_col (str): Event-B (next) start timestamp column.
    next_end_col (str): Event-B (next) end timestamp column.
    prev_start_col (Optional[str]): Previous-event start column, for consecutive mode.
    prev_end_col (Optional[str]): Previous-event end column, for consecutive mode.
    output_columns (Optional[List[str]]): Subset of computed columns to keep. Available:
        ``Has_Overlap, Overlap_Duration, Overlap_Start, Overlap_End, Merged_Start,
        Merged_End, Time_Gap, Event_Relationship, Overlap_Percentage``. When None, all are kept.
    time_unit (str): Unit for duration and gap columns: ``seconds``, ``minutes``, ``hours``, ``days``.
    adjacency_tolerance (int): Maximum ``Time_Gap`` (in ``time_unit``) still labelled ``ADJACENT``.

    Returns:
    DataFrame: Input DataFrame plus the computed overlap columns (or the subset in
        ``output_columns``).

    """
    consecutive_mode = prev_start_col is not None and prev_end_col is not None

    time_multipliers = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}
    multiplier = time_multipliers.get(time_unit, 1)

    df_result = df

    cols_to_convert = [start_time_col, end_time_col, next_start_col, next_end_col]
    if consecutive_mode:
        cols_to_convert.extend([prev_start_col, prev_end_col])

    for col_name in cols_to_convert:
        if col_name in df.columns:
            df_result = df_result.withColumn(f"_{col_name}_unix", F.unix_timestamp(F.col(col_name)))

    start_unix = F.col(f"_{start_time_col}_unix")
    end_unix = F.col(f"_{end_time_col}_unix")
    next_start_unix = F.col(f"_{next_start_col}_unix")
    next_end_unix = F.col(f"_{next_end_col}_unix")

    if consecutive_mode:
        prev_start_unix = F.col(f"_{prev_start_col}_unix")
        prev_end_unix = F.col(f"_{prev_end_col}_unix")

    # Has_Overlap
    if consecutive_mode:
        df_result = df_result.withColumn(
            "Has_Overlap",
            F.when(
                next_start_unix.isNull(),
                F.when(prev_start_unix.isNull(), F.lit(False))
                .when((end_unix > prev_start_unix) & (prev_end_unix > start_unix), F.lit(True))
                .otherwise(F.lit(False)),
            )
            .when((end_unix > next_start_unix) & (next_end_unix > start_unix), F.lit(True))
            .otherwise(F.lit(False)),
        )
    else:
        df_result = df_result.withColumn(
            "Has_Overlap",
            F.when(next_start_unix.isNull(), F.lit(False))
            .when((end_unix > next_start_unix) & (next_end_unix > start_unix), F.lit(True))
            .otherwise(F.lit(False)),
        )

    # Overlap_Duration
    if consecutive_mode:
        df_result = df_result.withColumn(
            "Overlap_Duration",
            F.when(
                F.col("Has_Overlap") & next_start_unix.isNotNull(),
                (F.least(end_unix, next_end_unix) - F.greatest(start_unix, next_start_unix))
                / multiplier,
            )
            .when(
                F.col("Has_Overlap") & prev_start_unix.isNotNull(),
                (F.least(end_unix, prev_end_unix) - F.greatest(start_unix, prev_start_unix))
                / multiplier,
            )
            .otherwise(F.lit(0)),
        )
    else:
        df_result = df_result.withColumn(
            "Overlap_Duration",
            F.when(
                F.col("Has_Overlap"),
                (F.least(end_unix, next_end_unix) - F.greatest(start_unix, next_start_unix))
                / multiplier,
            ).otherwise(F.lit(0)),
        )

    # Overlap_Start
    if consecutive_mode:
        df_result = df_result.withColumn(
            "Overlap_Start",
            F.when(
                F.col("Has_Overlap") & next_start_unix.isNotNull(),
                F.greatest(start_unix, next_start_unix),
            )
            .when(
                F.col("Has_Overlap") & prev_start_unix.isNotNull(),
                F.greatest(start_unix, prev_start_unix),
            )
            .otherwise(F.lit(None)),
        )
    else:
        df_result = df_result.withColumn(
            "Overlap_Start",
            F.when(F.col("Has_Overlap"), F.greatest(start_unix, next_start_unix)).otherwise(
                F.lit(None)
            ),
        )

    # Overlap_End
    if consecutive_mode:
        df_result = df_result.withColumn(
            "Overlap_End",
            F.when(
                F.col("Has_Overlap") & next_start_unix.isNotNull(), F.least(end_unix, next_end_unix)
            )
            .when(
                F.col("Has_Overlap") & prev_start_unix.isNotNull(), F.least(end_unix, prev_end_unix)
            )
            .otherwise(F.lit(None)),
        )
    else:
        df_result = df_result.withColumn(
            "Overlap_End",
            F.when(F.col("Has_Overlap"), F.least(end_unix, next_end_unix)).otherwise(F.lit(None)),
        )

    # Time_Gap
    if consecutive_mode:
        df_result = df_result.withColumn(
            "Time_Gap",
            F.when(F.col("Has_Overlap"), F.lit(0))
            .when(
                next_start_unix.isNotNull(),
                F.greatest(
                    (next_start_unix - end_unix) / multiplier,
                    (start_unix - next_end_unix) / multiplier,
                ),
            )
            .when(
                prev_start_unix.isNotNull(),
                F.greatest(
                    (start_unix - prev_end_unix) / multiplier,
                    (prev_start_unix - end_unix) / multiplier,
                ),
            )
            .otherwise(F.lit(None)),
        )
    else:
        df_result = df_result.withColumn(
            "Time_Gap",
            F.when(F.col("Has_Overlap"), F.lit(0))
            .when(next_start_unix.isNull(), F.lit(None))
            .otherwise(
                F.greatest(
                    (next_start_unix - end_unix) / multiplier,
                    (start_unix - next_end_unix) / multiplier,
                )
            ),
        )

    # Merged_Start
    if consecutive_mode:
        df_result = df_result.withColumn(
            "Merged_Start",
            F.when(
                (F.col("Has_Overlap") | (F.col("Time_Gap") <= adjacency_tolerance))
                & next_start_unix.isNotNull(),
                F.least(start_unix, next_start_unix),
            )
            .when(
                (F.col("Has_Overlap") | (F.col("Time_Gap") <= adjacency_tolerance))
                & prev_start_unix.isNotNull(),
                F.least(start_unix, prev_start_unix),
            )
            .otherwise(F.lit(None)),
        )
    else:
        df_result = df_result.withColumn(
            "Merged_Start",
            F.when(
                (F.col("Has_Overlap") | (F.col("Time_Gap") <= adjacency_tolerance))
                & next_start_unix.isNotNull(),
                F.least(start_unix, next_start_unix),
            ).otherwise(F.lit(None)),
        )

    # Merged_End
    if consecutive_mode:
        df_result = df_result.withColumn(
            "Merged_End",
            F.when(
                (F.col("Has_Overlap") | (F.col("Time_Gap") <= adjacency_tolerance))
                & next_start_unix.isNotNull(),
                F.greatest(end_unix, next_end_unix),
            )
            .when(
                (F.col("Has_Overlap") | (F.col("Time_Gap") <= adjacency_tolerance))
                & prev_start_unix.isNotNull(),
                F.greatest(end_unix, prev_end_unix),
            )
            .otherwise(F.lit(None)),
        )
    else:
        df_result = df_result.withColumn(
            "Merged_End",
            F.when(
                (F.col("Has_Overlap") | (F.col("Time_Gap") <= adjacency_tolerance))
                & next_start_unix.isNotNull(),
                F.greatest(end_unix, next_end_unix),
            ).otherwise(F.lit(None)),
        )

    # Event_Relationship
    if consecutive_mode:
        df_result = df_result.withColumn(
            "Event_Relationship",
            F.when(
                next_start_unix.isNull(),
                F.when(prev_start_unix.isNull(), F.lit("SINGLE_EVENT"))
                .when(F.col("Has_Overlap"), F.lit("OVERLAP"))
                .when(F.col("Time_Gap") <= adjacency_tolerance, F.lit("ADJACENT"))
                .when(F.col("Time_Gap") > 0, F.lit("GAP"))
                .otherwise(F.lit("UNKNOWN")),
            )
            .when(F.col("Has_Overlap"), F.lit("OVERLAP"))
            .when(F.col("Time_Gap") <= adjacency_tolerance, F.lit("ADJACENT"))
            .when(F.col("Time_Gap") > 0, F.lit("GAP"))
            .otherwise(F.lit("UNKNOWN")),
        )
    else:
        df_result = df_result.withColumn(
            "Event_Relationship",
            F.when(next_start_unix.isNull(), F.lit("SINGLE_EVENT"))
            .when(F.col("Has_Overlap"), F.lit("OVERLAP"))
            .when(F.col("Time_Gap") <= adjacency_tolerance, F.lit("ADJACENT"))
            .when(F.col("Time_Gap") > 0, F.lit("GAP"))
            .otherwise(F.lit("UNKNOWN")),
        )

    # Overlap_Percentage
    df_result = df_result.withColumn(
        "Overlap_Percentage",
        F.when(
            F.col("Has_Overlap"),
            (F.col("Overlap_Duration") * multiplier / (end_unix - start_unix)) * 100,
        ).otherwise(F.lit(0)),
    )

    df_result = df_result.withColumn(
        "Time_Gap",
        F.when(next_start_unix.isNull(), F.lit(None)).otherwise(F.col("Time_Gap")),
    )

    temp_cols = [f"_{c}_unix" for c in cols_to_convert]
    df_result = df_result.drop(*[c for c in temp_cols if c in df_result.columns])

    if output_columns:
        calculated_columns = [
            "Has_Overlap",
            "Overlap_Duration",
            "Overlap_Start",
            "Overlap_End",
            "Merged_Start",
            "Merged_End",
            "Time_Gap",
            "Event_Relationship",
            "Overlap_Percentage",
        ]
        original_columns = [c for c in df.columns if c not in calculated_columns]
        cols_to_keep = original_columns + [c for c in output_columns if c in calculated_columns]
        df_result = df_result.select(*cols_to_keep)

    return df_result


def merge_events(
    df: DataFrame,
    group_by_cols: list[str],
    order_col: str,
    end_col: str,
    event_relationship_col: str = "Event_Relationship",
    overlap_values: list[str] | None = None,
    agg_first_cols: dict[str, str] | None = None,
    agg_exprs_lst: list | None = None,
    next_event_columns: list[str] | None = None,
    ordered_ids: bool = False,
) -> DataFrame:
    """Collapse consecutive overlapping / adjacent events into single merged rows.

    Expects the input DataFrame to have a start-time column (``order_col``), an end-time column
    (``end_col``), and a relationship column identifying which consecutive pairs to merge
    (see :func:`analyze_temporal_overlaps`).

    Parameters:
    df (DataFrame): Input Spark DataFrame with per-event rows.
    group_by_cols (List[str]): Partition columns; events are only merged within the same
        partition (e.g. per entity / device / user).
    order_col (str): Chronological ordering column within each partition (the event start time).
    end_col (str): Event end-time column used for duration and range aggregation.
    event_relationship_col (str): Column whose value identifies mergeable relationships.
    overlap_values (Optional[List[str]]): Relationship values that trigger merging. Defaults
        to ``["OVERLAP", "ADJACENT"]``.
    agg_first_cols (Optional[Dict[str, str]]): Mapping ``{source_col: alias}`` for first()
        aggregations on merged groups. Defaults to ``{order_col: order_col}``.
    agg_exprs_lst (Optional[list]): Extra Spark aggregation expressions to append.
    next_event_columns (Optional[List[str]]): Columns for which to add ``Next_*`` lead
        columns after merging. Defaults to ``[order_col, end_col]``. Must include ``order_col``
        (used for the ``Time_to_next_event_sec`` gap column).
    ordered_ids (bool): When True, ``Group_ID`` is a sequential dense rank; otherwise it is
        a fast ``xxhash64`` (recommended for large data).

    Returns:
    DataFrame: The input columns plus ``Group_ID``, ``Grouped_Events_Count``, ``Next_*``
        lead columns, ``Time_to_next_event_sec`` and (for merged rows) ``Event_Duration_sec``.

    """
    if overlap_values is None:
        overlap_values = ["OVERLAP", "ADJACENT"]
    if agg_first_cols is None:
        agg_first_cols = {order_col: order_col}
    if agg_exprs_lst is None:
        agg_exprs_lst = []
    if next_event_columns is None:
        next_event_columns = [order_col, end_col]

    df_sorted = df.orderBy(*group_by_cols, order_col)
    window_ordered = Window.partitionBy(*group_by_cols).orderBy(order_col)

    df_with_prev = df_sorted.withColumn(
        "prev_Event_Relationship", F.lag(event_relationship_col).over(window_ordered)
    )

    df_with_flag = df_with_prev.withColumn(
        "group_change",
        F.when(
            (F.col(event_relationship_col) != F.col("prev_Event_Relationship"))
            | F.col("prev_Event_Relationship").isNull(),
            1,
        ).otherwise(0),
    )

    df = df_with_flag.withColumn(
        "Local_Group_ID",
        F.sum("group_change").over(
            window_ordered.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        ),
    )

    if ordered_ids:
        window_for_global_id = Window.orderBy(*group_by_cols, "Local_Group_ID")
        df = df.withColumn("Group_ID", F.dense_rank().over(window_for_global_id))
    else:
        df = df.withColumn(
            "Group_ID",
            F.xxhash64(F.concat_ws("||", *group_by_cols, F.col("Local_Group_ID"))),
        )

    df = df.drop("Local_Group_ID", "prev_Event_Relationship", "group_change")

    if agg_first_cols or agg_exprs_lst:
        idx_condition = None
        for value in overlap_values:
            cond = F.col(event_relationship_col) == value
            idx_condition = cond if idx_condition is None else (idx_condition | cond)

        df_to_aggregate = df.filter(idx_condition)
        df_no_aggregate = df.filter(~idx_condition)

        has_aggregate_data = df_to_aggregate.count() > 0
        has_no_aggregate_data = df_no_aggregate.count() > 0

        if has_aggregate_data:
            agg_exprs = []
            for source_col, alias_col in agg_first_cols.items():
                agg_exprs.append(F.first(source_col).alias(alias_col))
            for agg in agg_exprs_lst:
                agg_exprs.append(agg)

            endtime_condition = None
            for value in overlap_values:
                cond = F.first(event_relationship_col) == value
                endtime_condition = (
                    cond if endtime_condition is None else (endtime_condition | cond)
                )

            agg_exprs.append(
                F.when(endtime_condition, F.max(end_col)).otherwise(F.first(end_col)).alias(end_col)
            )
            agg_exprs.append(F.first(event_relationship_col).alias(event_relationship_col))
            agg_exprs.append(F.count("*").alias("Grouped_Events_Count"))

            grouped_sp = df_to_aggregate.groupBy(*group_by_cols, "Group_ID").agg(*agg_exprs)
            grouped_sp = grouped_sp.withColumn(
                "Event_Duration_sec",
                F.unix_timestamp(F.col(end_col)) - F.unix_timestamp(F.col(order_col)),
            )
            result_aggregated = grouped_sp
        else:
            result_aggregated = None

        if has_no_aggregate_data:
            result_no_aggregate = df_no_aggregate.withColumn("Grouped_Events_Count", F.lit(1))
        else:
            result_no_aggregate = None

        if result_aggregated is not None and result_no_aggregate is not None:
            result_sp = result_aggregated.unionByName(result_no_aggregate, allowMissingColumns=True)
        elif result_aggregated is not None:
            result_sp = result_aggregated
        elif result_no_aggregate is not None:
            result_sp = result_no_aggregate
        else:
            result_sp = df.limit(0)
    else:
        result_sp = df

    window_spec = Window.partitionBy(*group_by_cols).orderBy(F.col(order_col))
    next_event_cols = [f"Next_{ucol}" for ucol in next_event_columns]
    for orig_col, next_col in zip(next_event_columns, next_event_cols, strict=False):
        result_sp = result_sp.withColumn(next_col, F.lead(F.col(orig_col), 1).over(window_spec))

    result_sp = result_sp.withColumn(
        "Time_to_next_event_sec",
        (F.unix_timestamp(F.col(f"Next_{order_col}")) - F.unix_timestamp(F.col(end_col))).cast(
            "int"
        ),
    )

    original_columns = df.columns
    existing_cols = [c for c in original_columns if c in result_sp.columns]
    new_cols = [c for c in result_sp.columns if c not in original_columns]
    result_sp = result_sp.select(*(existing_cols + new_cols))

    result_sp = result_sp.orderBy(*group_by_cols, "Group_ID", order_col).withColumn(
        "row_id", F.monotonically_increasing_id()
    )

    return result_sp
