import dsToolbox.io_funcs as io_funcs
import dsToolbox.common_funcs as cfuncs

import pandas as pd
import datetime as dt

import pyspark.sql.functions as F
import pyspark.sql.types as spk_dtp
from pyspark.sql.window import Window

from pyspark.sql import DataFrame as DataFrame_ps
from typing import List

def asof_join_sub(l, r, left_on, right_on, left_by, right_by,
              tolerance=pd.Timedelta('600S'),
               direction='forward',
              **kwargs
             ):
    l[left_on] = pd.to_datetime(l[left_on])
    r[right_on] = pd.to_datetime(r[right_on])
    
    l = l.sort_values(left_on)
    r = r.sort_values(right_on)
    r = r.dropna(subset=[right_on])
    return pd.merge_asof(l, r,
                         left_on=left_on,
                         right_on=right_on,
                         left_by=left_by,
                         right_by=right_by,
                         tolerance=tolerance,
                         direction=direction,
                        **kwargs)
    
def asof_join_spark2(df_left,
                    df_right,
                    left_on,
                    right_on,
                    left_by,
                    right_by,
                    tolerance=pd.Timedelta('600S'),
                    direction='forward',
                    **kwargs
                ):
    '''
    '''
    # df_left=df_cycle_features
    # df_right=df_mcs
    # left_on='macro_cycle_endtime'
    # right_on='LoadFullTimeStamp'
    # by="equipment_id"
    
    common_cols = list(set(df_left.columns).intersection(df_right.columns))
    print(f"Common_cols = {common_cols}")
    
    for col in common_cols:
      if 'suffixes' in kwargs:
          df_left = df_left.withColumnRenamed(col, col+kwargs['suffixes'][0])
          df_right = df_right.withColumnRenamed(col, col+kwargs['suffixes'][1])
          if col==left_on:
            left_on = col+kwargs['suffixes'][0]
          if col==right_on:
            right_on = col+kwargs['suffixes'][1]
          if col==left_by:
            left_by = col+kwargs['suffixes'][0]
          if col==right_by:
            right_by = col+kwargs['suffixes'][1]
      
      else:
          df_left = df_left.withColumnRenamed(col, col+'_left')
          df_right = df_right.withColumnRenamed(col, col+'_right')
          if col==left_on:
            left_on = col+'_left'
          if col==right_on:
            right_on = col+'_right'
          if col==left_by:
            left_by = col+'_left'
          if col==right_by:
            right_by = col+'_right'
            
    schema_left = [i for i in df_left.schema]
    schema_right = [i for i in df_right.schema]

    NewSchema = spk_dtp.StructType(schema_left+schema_right)
    df_left.sort(left_by,left_on)
    df_right.sort(right_by,right_on)

    def asof_join_wrapped(l, r):
        return asof_join_sub(l, r, left_on, right_on, left_by, right_by,
                         tolerance=tolerance,
                         direction=direction, **kwargs)
    
    left_grp = df_left.groupby(left_by)
    right_grp = df_right.groupby(right_by)
    df_joined =left_grp.cogroup(right_grp).applyInPandas(asof_join_wrapped, schema=NewSchema)
    return df_joined

def melt(df: DataFrame_ps,
        id_vars: List[str], value_vars: List[str],
        var_name: str="variable", value_name: str="value") -> DataFrame_ps:
    '''
    Implements the equivalent of pandas melt (https://pandas.pydata.org/docs/reference/api/pandas.melt.html) in PySpark.
    The implementation is based on https://stackoverflow.com/questions/41670103/how-to-melt-spark-dataframe
    Notes: The implementation can be improved to accept None for id_vars or value_vars as the pandas melt function
    :param df: starting PySpark data frame in wide format
    :param id_vars: Column(s) to use as identifier variables.
    :param value_vars: Column(s) to unpivot. If not specified, uses all columns that are not set as id_vars.
    :param var_name: Name to use for the ‘variable’ column.
    :param value_name: Name to use for the ‘value’ column.
    :return: PySpark data frame in long format
    '''

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

def rename_cols(sp, mapCols_dict):
  for old_name, new_name in mapCols_dict.items():
    sp = sp.withColumnRenamed(old_name, new_name)
  return sp

def sp_to_numeric(sp, exclude_cols, caseTo='float'):
  import pyspark.sql.functions as F
  non_str_cols=[col for col in  sp.columns if col not in exclude_cols]
  for ucol in non_str_cols:
    sp=sp.withColumn(ucol,F.col(ucol).cast(caseTo))
  return sp

def col_finder(key_vault_dict,
              tableName = 'mcsdata.mcs_bm_15',
              cols2search=['facies_','formation_'],
              ):
  df_cols=io_funcs.query_deltaTable_db(
                                      f'SHOW COLUMNS IN {tableName};',
                                      key_vault_dict=key_vault_dict,
                                      verbose=False
                                      ).toPandas().squeeze().tolist()

  col_finder, _=cfuncs.inWithReg(cols2search, df_cols)
  return col_finder

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -----------------------Update a delta table  or blob---------------------
def last_date(output_name,
              date_col='date',
              custom_config=None,
              key_vault_dict='deltaTable',   ###for only delta_table
              platform='databricks',         ### for only blob
              ):
  
  if (isinstance(output_name,str)) and (io_funcs.deltaTable_check(output_name)): 
    saved_dates = io_funcs.query_deltaTable_db(
                                              f"""select min({date_col}) as min_time ,
                                                          max({date_col}) as max_time
                                                  from   {output_name}""",
                                              key_vault_dict=key_vault_dict,
                                              custom_config=custom_config,
                                              verbose=False 
                                              ).toPandas()  
    last_save_date=saved_dates['max_time'].iloc[0]
    print(f"The last date found in delta_table:{last_save_date}")

  elif (isinstance(output_name, dict)) and (io_funcs.blob_check(blob_dict=output_name,
                                                                custom_config=custom_config,
                                                                platform=platform,)):
    udata = io_funcs.blob2pd(blob_dict=output_name,
                            custom_config=custom_config,
                            platform=platform,
                            #  **kwargs_csv,
                            )
    udata[date_col] = pd.to_datetime(udata[date_col], format="%Y-%m-%d")
    last_save_date= udata[date_col].max()  
    
    print(f"The last date found in blob:{last_save_date}")

  else:
    last_save_date=None

  return last_save_date

def save_outputs(ouputs_dict_list,
                **kwargs,
              ):
  import inspect
  import dsToolbox.io_funcs as io_funcs
  
  spark2del_args = list(inspect.signature(io_funcs.spark2deltaTable).parameters)
  spark2del_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in spark2del_args}
  pd2blob_args = list(inspect.signature(io_funcs.pd2blob).parameters)
  pd2blob_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in pd2blob_args}

  outputs=ouputs_dict_list.items() if isinstance(ouputs_dict_list, dict) else ouputs_dict_list

  for (tableName_blobDict , sp) in outputs:
    print(tableName_blobDict)
    if isinstance(tableName_blobDict, str): 
      io_funcs.spark2deltaTable(
                                sp,
                                table_name    = tableName_blobDict.split('.')[1],
                                schema        = tableName_blobDict.split('.')[0],
                                write_mode = 'append',
                                mergeSchema=True,
                                **spark2del_args
                              )
    elif isinstance(tableName_blobDict, dict): 
      io_funcs.pd2blob(sp,
                      blob_dict=tableName_blobDict,
                      overwrite=False,
                      append=True,
                      **pd2blob_args
                      )

##TODO: update based on run_recursively
def update_db_recursively(dfGenerator_func,
                          output_name,
                          year_range=[2021,2099],
                          firstDate=None,
                          lastDate=dt.datetime.now().date(),
                          date_col='date',
                          custom_config=None,
                          key_vault_dict='deltaTable',   ###for  delta_table only
                          platform='databricks',         ### for blob only
                          **kwargs
                          ):
  """
  Run a function recursively to create each month's results and append them to a delta table/blob storage
  
  The first step creates a list of months (LOM) which must be run recursively. 
  The last LOM date will be set to running code date. 
  If there is already an output, it will use the last day of the output to modify the first date of LOM. 
  After generating a list of dates, the function runs and appends the results to output
  For example with the following inputs:
    year_range=[2021,2099]
    the last date of record in output deltatable: '2023-15-07'  
    run_date: '2023-22-12'

  The list of months will be:
  ['2023-16-07', '2021-01-03', '2021-01-04',...,'2023-22-11']  
  and function will be run recursively:

  from '2021-16-07'  to '2021-31-07' and append the result to output
  from '2021-01-08'  to '2021-31-08' and append the result to output
  ...
  from '2023-01-12'  to '2021-22-11' and append the result to output

  
    Params:
      dfGenerator_func:(function)  Define a function that generates spark|panadas dataframe based on three main arguments: start_date, end_date, output_name
                                                          
      output_name:(string or dictionary) : the name of output, if it is a string , it is a delta table and if it is a dinctonary it is a blob file
        
      year_range(list): A list in [first_year, last_year] format. The first and last years are used to create a list of dates from the first day of the first month of the first year. For example for [2021, 2099] , it creates the following list ['2021-01-01', '2021-01-02, ....,'2099-31-12'] . 
      
      firstDate: the first date of range , regardless of the last date in output
      lastDate (dt.datetime.now().date()):  the last date of range
                          
      custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead
      
      key_vault_dict='deltaTable': option only for delta table output see spark2deltaTable function for more information

      platform='databricks'  :  option only for blob storage output  see pd2blob function for more information
      

    Returns:
            df - Dataframe with rolling features        
  """  
  import inspect
  import dsToolbox.io_funcs as io_funcs
  
  spark2del_args = list(inspect.signature(io_funcs.spark2deltaTable).parameters)
  pd2blob_args = list(inspect.signature(io_funcs.pd2blob).parameters)
  dfGenerator_func_args = {k: kwargs.pop(k) for k in dict(kwargs) if k not in (spark2del_args+pd2blob_args)}
  
  import datetime as dt  

  last_save_date=last_date(                     ###for  delta_table only
                          output_name,
                          date_col=date_col,
                          custom_config=custom_config,
                          key_vault_dict=key_vault_dict,   ###for  delta_table only
                          platform=platform,         ### for blob only
                          )
  warn_txt=False
  if (firstDate is not None):
    if isinstance(firstDate, str):
      print(firstDate)
      firstDate   = dt.datetime.strptime(firstDate, "%Y-%m-%d").date()
    elif isinstance(firstDate,pd._libs.tslibs.timestamps.Timestamp):
      firstDate   =firstDate.date()
    warn_txt=True  
  else:
    firstDate= None if last_save_date is None else last_save_date+dt.timedelta(days=1)
  
  ###polish the warning, it is meaningless when last_save_date exists
  if (warn_txt)& (last_save_date is not None):
    print(f"last date is {last_save_date}; however, the function starts from given first date: {firstDate}")

  udates=cfuncs.datesList(year_range=year_range, 
                          firstDate=firstDate,
                          lastDate=lastDate
                          ###'2020-01-01'
                          )
  
  if len(udates)==0:
    print("Database|file is updated")
  else:
    print("date list updated to :\n", udates)

  for ii in range(len(udates)-1):     
    try: 
      ##TODO: what if, there are more than on output, in that case output_name is only for the first output, 
      start_date, end_date= cfuncs.extract_start_end(udates, ii)
      ouputs_list=dfGenerator_func(start_date, end_date,
                                  output_name,
                                  **dfGenerator_func_args
                                  )
      
      save_outputs(###for  delta_table only
                   ouputs_list)
      
    except Exception as e:
      print(f'***Creating Database(s) for {start_date} failed: \n\t\t {str(e)}')
      print('**********************************************************************************************')
      ##sys.exit()

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -----------------------------------------fEng----------------------------
def create_rolling_features(df, timestamp_col_name: str = 'Time_Stamp', groupby_col_name: str = 'machine',
                            window_duration: str = '5 minutes', agg_type: str = 'avg'):
  
  """
  Create rolling features for numeric column in the dataframe. Rolling features are not created on
  columns with dtypes 'date', 'string', 'timestamp'.
  
    Inputs:
            df - Dataframe
            timestamp_col_name - Time-stamp column name. Rolling window creation is on this column.
            groupby_col_name -  It is use to partition "df" by this column and then do aggregation as
                                as timestamp column may be overlapping for "groupby_col_name". E.g. -
                                Different machines have entries at the same/overlapping timestamp.
            window_duration - Rolling window duration (in seconds)
            agg_type - Aggregation type (e.g.- 'avg', 'min', 'max', etc.)
            
    Returns:
            df - Dataframe with rolling features        
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
  
  w = Window.partitionBy(groupby_col_name).orderBy(F.col(timestamp_col_name).cast('long')).rangeBetween(-window_duration, 0)
  
  for column in numeric_cols:
    df = eval(f"df.withColumn('{column}_{agg_type}', F.{agg_type}('{column}').over(w))")
    df = eval(f"df.withColumn('{column}_{agg_type}', F.col('{column}_{agg_type}').cast('float'))")
    df = df.drop(column)
    
  return df

def create_tumbling_features(df, timestamp_col_name: str = 'Time_Stamp', groupby_col_name: str = 'machine',
                             window_duration: str = '5 minutes', agg_type: str = 'avg', direction: str = 'backward',
                             tolerance:str = None):
  
  """
  Create tumbling features for numeric column in the dataframe. Tumbling features are not created on
  columns with dtypes 'date', 'string', 'timestamp'.
  
    Inputs:
            df - Dataframe
            timestamp_col_name - Time-stamp column name. Tumbling window creation is on this column.
            groupby_col_name -  It is use to partition "df" by this column and then do aggregation as
                                as timestamp column may be overlapping for "groupby_col_name". E.g. -
                                Different machines have entries at the same/overlapping timestamp.
            window_duration - Tumbling window duration (e.g. - '5 minutes')
            agg_type - Aggregation type (e.g.- 'avg', 'min', 'max', etc.)
            direction - Whether to search for prior, subsequent, or closest matches ('backward', 'forward', 'nearest').
            tolerance - Select asof tolerance within this range.
            
    Returns:
            df - Dataframe with tumbling features        
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
  
  df_right = eval(f"df.groupBy(groupby_col_name, F.window(timestamp_col_name, window_duration)).{agg_type}()")
  df_right = df_right.withColumn('window_start', df_right.window.start).withColumn('window_end', df_right.window.end)
  df_right = df_right.drop('window')
  
  for column in numeric_cols:
    df_right = eval(f"df_right.withColumnRenamed('{agg_type}({column})', '{column}_{agg_type}')")
    df_right = eval(f"df_right.withColumn('{column}_{agg_type}', F.col('{column}_{agg_type}').cast('float'))")
  
  df_merged = asof_join_spark2(df_left, df_right, left_on=timestamp_col_name, right_on='window_start',
                              by=groupby_col_name, tolerance=pd.Timedelta(tolerance), direction=direction)
    
  return df_merged
