"""Spark geospatial helpers: Euclidean 2D/3D distance and Haversine great-circle distance."""

from typing import Literal, Optional, Tuple, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

_Coord2 = Union[Tuple[str, str], Tuple[float, float]]
_CoordN = Union[Tuple[str, ...], Tuple[float, ...]]


def _coord_exprs(coords: _CoordN):
    """Convert each element of ``coords`` to a Spark Column (col ref for str, literal for numeric)."""
    return tuple(F.col(c) if isinstance(c, str) else F.lit(float(c)) for c in coords)


def _apply_null_strategy(expr, null_conditions, null_strategy: str):
    """Apply ``skip`` (return null) or ``zero`` (return 0.0) when any coordinate is null."""
    null_condition = null_conditions[0]
    for cond in null_conditions[1:]:
        null_condition = null_condition | cond
    if null_strategy == "skip":
        return F.when(null_condition, F.lit(None)).otherwise(expr)
    if null_strategy == "zero":
        return F.when(null_condition, F.lit(0.0)).otherwise(expr)
    raise ValueError(f"Invalid null_strategy: {null_strategy!r}. Use 'skip' or 'zero'.")


def calculate_distance(
    df: DataFrame,
    point1_coords: _CoordN,
    point2_coords: _CoordN,
    distance_col_name: str = "distance",
    null_strategy: Literal["skip", "zero"] = "skip",
    precision: Optional[int] = None,
) -> DataFrame:
    """Add a Euclidean-distance column between two 2D or 3D points to a Spark DataFrame.

    Parameters:
    df (DataFrame): Input Spark DataFrame.
    point1_coords (Union[Tuple[str, ...], Tuple[float, ...]]): First point as either column
        names ``(x1, y1)`` / ``(x1, y1, z1)`` or fixed numeric coordinates.
    point2_coords (Union[Tuple[str, ...], Tuple[float, ...]]): Second point in the same shape.
    distance_col_name (str): Name of the added distance column.
    null_strategy (Literal["skip", "zero"]): ``"skip"`` returns null when any coordinate is
        null; ``"zero"`` returns 0.0.
    precision (Optional[int]): If set, round the distance to this many decimal places.

    Returns:
    DataFrame: Input DataFrame with the distance column appended.

    """
    if len(point1_coords) != len(point2_coords):
        raise ValueError("Both points must have the same number of dimensions")
    if len(point1_coords) not in (2, 3):
        raise ValueError("Only 2D (x,y) and 3D (x,y,z) coordinates are supported")

    p1 = _coord_exprs(point1_coords)
    p2 = _coord_exprs(point2_coords)

    squared_diffs = []
    null_conditions = []
    for a, b in zip(p1, p2):
        squared_diffs.append(F.pow(b - a, 2))
        null_conditions.extend([a.isNull(), b.isNull()])

    distance_expr = F.sqrt(sum(squared_diffs))
    final_expr = _apply_null_strategy(distance_expr, null_conditions, null_strategy)
    if precision is not None:
        final_expr = F.round(final_expr, precision)
    return df.withColumn(distance_col_name, final_expr)


def calculate_haversine_distance(
    df: DataFrame,
    point1_coords: _Coord2,
    point2_coords: _Coord2,
    distance_col_name: str = "distance_km",
    unit: Literal["km", "miles", "meters"] = "meters",
    null_strategy: Literal["skip", "zero"] = "skip",
    precision: Optional[int] = None,
) -> DataFrame:
    """Add a Haversine great-circle distance column for two ``(lat, lon)`` points.

    Parameters:
    df (DataFrame): Input Spark DataFrame.
    point1_coords (Union[Tuple[str, str], Tuple[float, float]]): First point as either column
        names ``(lat1, lon1)`` or fixed decimal-degree coordinates.
    point2_coords (Union[Tuple[str, str], Tuple[float, float]]): Second point in the same shape.
    distance_col_name (str): Name of the added distance column.
    unit (Literal["km", "miles", "meters"]): Output unit. Uses Earth's mean radius per unit.
    null_strategy (Literal["skip", "zero"]): ``"skip"`` returns null when any coordinate is
        null; ``"zero"`` returns 0.0.
    precision (Optional[int]): If set, round the distance to this many decimal places.

    Returns:
    DataFrame: Input DataFrame with the distance column appended.

    """
    if len(point1_coords) != 2 or len(point2_coords) != 2:
        raise ValueError("Geographic coordinates must be 2D (latitude, longitude)")

    radius = {"km": 6371.0, "miles": 3959.0, "meters": 6371000.0}
    if unit not in radius:
        raise ValueError(f"Invalid unit: {unit!r}. Use 'km', 'miles', or 'meters'.")

    lat1, lon1 = _coord_exprs(point1_coords)
    lat2, lon2 = _coord_exprs(point2_coords)

    null_conditions = [lat1.isNull(), lon1.isNull(), lat2.isNull(), lon2.isNull()]

    lat1_r, lon1_r = F.radians(lat1), F.radians(lon1)
    lat2_r, lon2_r = F.radians(lat2), F.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = F.pow(F.sin(dlat / 2), 2) + F.cos(lat1_r) * F.cos(lat2_r) * F.pow(F.sin(dlon / 2), 2)
    c = 2 * F.asin(F.sqrt(a))
    distance_expr = F.lit(radius[unit]) * c

    final_expr = _apply_null_strategy(distance_expr, null_conditions, null_strategy)
    if precision is not None:
        final_expr = F.round(final_expr, precision)
    return df.withColumn(distance_col_name, final_expr)
