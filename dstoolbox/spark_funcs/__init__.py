"""spark_funcs package: asof joins, wide-long reshape, column rename/cast, column finder, ETL pipelines, time-series feature engineering, data-quality diagnostics, binning, geospatial distance, temporal-event analysis, and spatial proximity grouping."""

from .joins import (
    asof_join_spark2,
)

from .reshape import (
    melt,
)

from .columns import (
    rename_cols,
    sp_to_numeric,
)

from .col_finder import (
    col_finder,
)

from .pipelines import (
    last_date,
    save_outputs,
    update_db_recursively,
)

from .features import (
    create_rolling_features,
    create_tumbling_features,
)

from .diagnostics import (
    percent_missing,
    find_duplicates,
    find_lost_records,
)

from .binning import (
    cut,
)

from .geo import (
    calculate_distance,
    calculate_haversine_distance,
)

from .events import (
    prepare_consecutive_events,
    analyze_temporal_overlaps,
    merge_events,
)

from .spatial import (
    group_by_proximity,
)
