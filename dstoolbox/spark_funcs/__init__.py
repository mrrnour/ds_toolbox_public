"""spark_funcs package: asof joins, wide-long reshape, column rename/cast, column finder, ETL pipelines, time-series feature engineering, data-quality diagnostics, binning, geospatial distance, temporal-event analysis, and spatial proximity grouping."""

from .binning import (
    cut,
)
from .col_finder import (
    col_finder,
)
from .columns import (
    rename_cols,
    sp_to_numeric,
)
from .diagnostics import (
    find_duplicates,
    find_lost_records,
    percent_missing,
)
from .events import (
    analyze_temporal_overlaps,
    merge_events,
    prepare_consecutive_events,
)
from .features import (
    create_rolling_features,
    create_tumbling_features,
)
from .geo import (
    calculate_distance,
    calculate_haversine_distance,
)
from .joins import (
    asof_join_spark2,
)
from .pipelines import (
    last_date,
    save_outputs,
    update_db_recursively,
)
from .reshape import (
    melt,
)
from .spatial import (
    group_by_proximity,
)
