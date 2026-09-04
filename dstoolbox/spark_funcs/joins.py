import pandas as pd
import pyspark.sql.types as spk_dtp


def _asof_join(
    l,
    r,
    left_on,
    right_on,
    left_by,
    right_by,
    tolerance=pd.Timedelta("600S"),
    direction="forward",
    **kwargs,
):
    """Per-group as-of join helper that runs ``pd.merge_asof`` on two pandas frames.

    Internal helper wrapped by :func:`asof_join_spark2` and called inside
    ``cogroup.applyInPandas``.

    Parameters
    ----------
    l, r : pandas.DataFrame
        Left and right frames for one group.
    left_on, right_on : str
        Time-key columns; coerced to ``datetime`` and sorted before merging.
    left_by, right_by : str
        Group-by columns (kept for API symmetry; the cogroup already
        partitions, so they don't change semantics here).
    tolerance : pandas.Timedelta, optional
        Maximum allowed time gap between matched rows. Default 600s.
    direction : {'backward', 'forward', 'nearest'}, optional
        ``merge_asof`` direction. Default ``'forward'``.
    **kwargs
        Forwarded to ``pd.merge_asof``.

    Returns
    -------
    pandas.DataFrame
        Result of ``pd.merge_asof``.
    """
    l[left_on] = pd.to_datetime(l[left_on])
    r[right_on] = pd.to_datetime(r[right_on])

    l = l.sort_values(left_on)
    r = r.sort_values(right_on)
    r = r.dropna(subset=[right_on])
    return pd.merge_asof(
        l,
        r,
        left_on=left_on,
        right_on=right_on,
        left_by=left_by,
        right_by=right_by,
        tolerance=tolerance,
        direction=direction,
        **kwargs,
    )


def asof_join_spark2(
    df_left,
    df_right,
    left_on,
    right_on,
    left_by,
    right_by,
    tolerance=pd.Timedelta("600S"),
    direction="forward",
    **kwargs,
):
    """As-of join two Spark DataFrames using ``cogroup.applyInPandas``.

    Renames any common columns (with ``_left`` / ``_right`` suffixes or a
    user-supplied ``suffixes`` pair) before merging so the joined schema
    is unambiguous, then delegates to :func:`_asof_join` per group.

    Parameters
    ----------
    df_left, df_right : pyspark.sql.DataFrame
        Frames to join.
    left_on, right_on : str
        Time-key columns.
    left_by, right_by : str
        Group-by columns. The cogroup partitions on these.
    tolerance : pandas.Timedelta, optional
        Maximum allowed time gap. Default 600s.
    direction : {'backward', 'forward', 'nearest'}, optional
        Default ``'forward'``.
    **kwargs
        Forwarded to ``pd.merge_asof``. ``suffixes`` (a 2-tuple) is also
        used to rename column collisions.

    Returns
    -------
    pyspark.sql.DataFrame
        Joined frame.
    """
    # df_left=df_events
    # df_right=df_readings
    # left_on='event_time'
    # right_on='reading_time'
    # by="asset_id"

    common_cols = list(set(df_left.columns).intersection(df_right.columns))
    print(f"Common_cols = {common_cols}")

    for col in common_cols:
        if "suffixes" in kwargs:
            df_left = df_left.withColumnRenamed(col, col + kwargs["suffixes"][0])
            df_right = df_right.withColumnRenamed(col, col + kwargs["suffixes"][1])
            if col == left_on:
                left_on = col + kwargs["suffixes"][0]
            if col == right_on:
                right_on = col + kwargs["suffixes"][1]
            if col == left_by:
                left_by = col + kwargs["suffixes"][0]
            if col == right_by:
                right_by = col + kwargs["suffixes"][1]

        else:
            df_left = df_left.withColumnRenamed(col, col + "_left")
            df_right = df_right.withColumnRenamed(col, col + "_right")
            if col == left_on:
                left_on = col + "_left"
            if col == right_on:
                right_on = col + "_right"
            if col == left_by:
                left_by = col + "_left"
            if col == right_by:
                right_by = col + "_right"

    schema_left = [i for i in df_left.schema]
    schema_right = [i for i in df_right.schema]

    NewSchema = spk_dtp.StructType(schema_left + schema_right)
    df_left.sort(left_by, left_on)
    df_right.sort(right_by, right_on)

    def asof_join_wrapped(l, r):
        """Closure passed to ``applyInPandas`` that captures the resolved keys."""
        return _asof_join(
            l,
            r,
            left_on,
            right_on,
            left_by,
            right_by,
            tolerance=tolerance,
            direction=direction,
            **kwargs,
        )

    left_grp = df_left.groupby(left_by)
    right_grp = df_right.groupby(right_by)
    df_joined = left_grp.cogroup(right_grp).applyInPandas(asof_join_wrapped, schema=NewSchema)
    return df_joined
