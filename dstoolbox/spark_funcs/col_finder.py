import dstoolbox.io_funcs as io_funcs
from dstoolbox import utils

import pandas as pd
import datetime as dt

import pyspark.sql.functions as F
import pyspark.sql.types as spk_dtp
from pyspark.sql.window import Window

from pyspark.sql import DataFrame as DataFrame_ps
from typing import List
def col_finder(key_vault_dict,
              table_name='schema.table_name',
              cols_to_search=None,
              ):
  """Look up columns in a Delta table that match any of the supplied regex patterns.

  Parameters
  ----------
  key_vault_dict : str
      Key Vault entry name passed to ``query_delta_table_db``.
  table_name : str, optional
      Fully qualified Delta table to inspect (``schema.table``).
      Default ``'schema.table_name'`` (placeholder — caller should override).
  cols_to_search : list of str or None, optional
      Regex patterns to search for. Defaults to ``['facies_', 'formation_']``.

  Returns
  -------
  list of str
      Column names from ``table_name`` that match any pattern.
  """
  if cols_to_search is None:
    cols_to_search = ['facies_', 'formation_']
  df_cols=io_funcs.query_delta_table_db(
                                      f'SHOW COLUMNS IN {table_name};',
                                      key_vault_dict=key_vault_dict,
                                      verbose=False
                                      ).toPandas().squeeze().tolist()

  matched_cols, _=utils.regex_filter_list(cols_to_search, df_cols)
  return matched_cols

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
