import dstoolbox.io_funcs as io_funcs
from dstoolbox import utils

import pandas as pd
import datetime as dt

import pyspark.sql.functions as F
import pyspark.sql.types as spk_dtp
from pyspark.sql.window import Window

from pyspark.sql import DataFrame as DataFrame_ps
from typing import List
# -------------------------------------------------------------------------
# -----------------------------------------fEng----------------------------
def create_rolling_features(df, timestamp_column: str = 'Time_Stamp', group_by_column: str = 'machine',
                            window_duration: str = '5 minutes', agg_type: str = 'avg'):
  
  """Create rolling-window aggregate features on numeric columns.

  Rolling features are not created on columns with dtypes ``date``,
  ``string``, or ``timestamp``.

  Parameters
  ----------
  df : pyspark.sql.DataFrame
      Input frame.
  timestamp_column : str, default ``'Time_Stamp'``
      Time-stamp column used as the window ordering axis.
  group_by_column : str, default ``'machine'``
      Partition column; rolling windows are computed per group (e.g.
      per machine) so that overlapping timestamps across groups do not
      mix.
  window_duration : str, default ``'5 minutes'``
      Rolling-window size expressed as ``"<n> <unit>"``
      (``seconds`` | ``minutes`` | ``hours`` | ``days``).
  agg_type : str, default ``'avg'``
      Spark aggregation to apply (``'avg'``, ``'min'``, ``'max'``, ...).

  Returns
  -------
  pyspark.sql.DataFrame
      Frame with one rolling-aggregate column per numeric input column.
  """
  
  numeric_cols = [item[0] for item in df.dtypes if \
                      ((item[1].startswith('int')) \
                       | (item[1].startswith('float')) \
                       | (item[1].startswith('long')) \
                       | (item[1].startswith('double')) )]
  
  # Determining window_duration in seconds
  num_part = int(window_duration.split(' ')[0])
  string_part = window_duration.split(' ')[1]
  
  if string_part == 'seconds':
    window_duration = num_part
  elif string_part == 'minutes':
    window_duration = num_part*60
  elif string_part == 'hours':
    window_duration = num_part*60*60  
  elif string_part == 'days':
    window_duration = num_part*24*60*60
  
  w = Window.partitionBy(group_by_column).orderBy(F.col(timestamp_column).cast('long')).rangeBetween(-window_duration, 0)
  
  for column in numeric_cols:
    agg_func = getattr(F, agg_type)
    df = df.withColumn(f"{column}_{agg_type}", agg_func(column).over(w))
    df = df.withColumn(f"{column}_{agg_type}", F.col(f"{column}_{agg_type}").cast("float"))
    df = df.drop(column)
    
  return df

def create_tumbling_features(df, timestamp_column: str = 'Time_Stamp', group_by_column: str = 'machine',
                             window_duration: str = '5 minutes', agg_type: str = 'avg', direction: str = 'backward',
                             tolerance:str = None):
  
  """Create non-overlapping (tumbling) window aggregate features on numeric columns.

  Tumbling features are not created on columns with dtypes ``date``,
  ``string``, or ``timestamp``.

  Parameters
  ----------
  df : pyspark.sql.DataFrame
      Input frame.
  timestamp_column : str, default ``'Time_Stamp'``
      Time-stamp column used as the window axis.
  group_by_column : str, default ``'machine'``
      Partition column; tumbling windows are computed per group so that
      overlapping timestamps across groups do not mix.
  window_duration : str, default ``'5 minutes'``
      Tumbling window size expressed as ``"<n> <unit>"``.
  agg_type : str, default ``'avg'``
      Spark aggregation to apply.
  direction : {'backward', 'forward', 'nearest'}, default ``'backward'``
      Asof-join direction for attaching each row to its window.
  tolerance : str, optional
      Asof tolerance. Defaults to ``window_duration`` when unset.

  Returns
  -------
  pyspark.sql.DataFrame
      Original non-numeric columns merged with per-window aggregates.
  """
  
  if tolerance is None:
    tolerance = window_duration
  
  numeric_cols = [item[0] for item in df.dtypes if \
                      ((item[1].startswith('int')) \
                       | (item[1].startswith('float')) \
                       | (item[1].startswith('long')) \
                       | (item[1].startswith('double')) )]
  
  non_numeric_cols = [item[0] for item in df.dtypes if not\
                      ((item[1].startswith('int')) \
                       | (item[1].startswith('float')) \
                       | (item[1].startswith('long')) \
                       | (item[1].startswith('double')) )]
  
  df_left = df.select(non_numeric_cols)
  
  grouped = df.groupBy(group_by_column, F.window(timestamp_column, window_duration))
  df_right = getattr(grouped, agg_type)()
  df_right = df_right.withColumn('window_start', df_right.window.start).withColumn('window_end', df_right.window.end)
  df_right = df_right.drop('window')
  
  for column in numeric_cols:
    df_right = df_right.withColumnRenamed(f"{agg_type}({column})", f"{column}_{agg_type}")
    df_right = df_right.withColumn(f"{column}_{agg_type}", F.col(f"{column}_{agg_type}").cast("float"))
  
  df_merged = asof_join_spark2(df_left, df_right, left_on=timestamp_column, right_on='window_start',
                              by=group_by_column, tolerance=pd.Timedelta(tolerance), direction=direction)
    
  return df_merged
