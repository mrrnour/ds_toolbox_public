import os,sys
import datetime as dt  
import pandas as pd
config_file=os.path.join('dsToolbox','config.yml')
import dsToolbox.common_funcs    as cfuncs

class cred_setup_mssql():
    def __init__(self, config_file):
        import yaml
        from importlib import resources as res
        with open(config_file, 'r') as stream:
            self.config = yaml.safe_load(stream)

    def MSSQL_connector__pyodbc(self):
        import pyodbc 
        MSSQL_config=   self.config['sql_server']
        cnxn = pyodbc.connect("DRIVER={SQL Server Native Client 11.0};"
                            f"SERVER={MSSQL_config['db_server']};"
                            "Trusted_Connection=yes;"
                            )
        return cnxn, MSSQL_config

    def MSSQL_connector__sqlalchemy(self):
        from sqlalchemy import create_engine
        import urllib
        MSSQL_config=   self.config['sql_server']

        # db_params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+MSSQL_config['db_server']+';Trusted_Connection=yes;')
        db_params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+MSSQL_config['db_server']+';DATABASE='+MSSQL_config['db_name']+';Trusted_Connection=yes;')

        engine = create_engine(f'mssql+pyodbc:///?odbc_connect={db_params}')
        
        return engine, db_params
    
    def incorta_connector(self):
        ###under edition: not tested yet
        import pyodbc 
        incorta_config = self.config['incorta_server']
        db_params = {
            "host"    : incorta_config['host'],
            "port"    : incorta_config['port'],
            "database": incorta_config['database'],
            "user"    : incorta_config['user'],
            "password": incorta_config['password']
        }    
        cnxn = pyodbc.connect(**db_params)

        return cnxn, db_params

def mSql_query(sql_query, config_file=config_file, return_df=True):
    import pandas as pd
    try:
        cnxn, _=cred_setup_mssql(config_file=config_file).MSSQL_connector__pyodbc()
        ##or using cursor:
        # cursor.execute(sql_query)
        # tables = cursor.fetchall()
        # for table in tables:
        #     print(table[0])

        # cnxn.execute(sql_query): This method is used to execute a SQL query against the database. The sql_query is a string that contains the SQL query you want to execute. This method does not return the results of the query. It is often used for SQL commands like INSERT, UPDATE, DELETE, etc., which modify the data but do not return any results.

        # cursor.fetchall(): This method is used to fetch all the rows of a query result. It returns a list of tuples where each tuple corresponds to a row in the result. You would typically use this method after executing a SELECT query to retrieve the data that the query returns.
    
        if return_df:
            out=pd.read_sql_query(sql_query, cnxn)
        else:
            out=cnxn.execute(sql_query)
    except Exception as e:
        cnxn.rollback()
        sys.exit("Error in running SQL in MS Sql Server: \n" + str(e))
    return out

def incorta_query(sql_query, config_file=config_file, return_df=True):
    ###under edition: not tested yet
    import pandas as pd
    try:
        cnxn, _=cred_setup_mssql(config_file=config_file).incorta_connector()
        if return_df:
            out=pd.read_sql_query(sql_query, cnxn)
        else:
            out=cnxn.execute(sql_query)
    except Exception as e:
        cnxn.rollback()
        sys.exit("Error in running SQL in INCORTA: \n" + str(e))

    return out

def df2MSQL(df, table_name, config_file=config_file, **kwargs):
    try:
        engine, _=cred_setup_mssql(config_file=config_file).MSSQL_connector__sqlalchemy()
        df.to_sql(table_name, con = engine,
                    **kwargs)

    except Exception as e:
        # engine.rollback()
        sys.exit("Error in writing dataFrame into MSSQL: \n" + str(e))

    return None

def MSql_table_check(tablename,
              config_file=config_file,
              ):
    import pandas as pd
    try:
        cnxn, _=cred_setup_mssql(config_file=config_file).MSSQL_connector__pyodbc()

        if tablename.count(".")==3:
            database, schema, tablename=tablename.split(".")[0], tablename.split(".")[1], tablename.split(".")[2]
            information_schema=f"{database}.information_schema.tables"
        else:
            schema, tablename=tablename.split(".")[0], tablename.split(".")[1]
            information_schema=f"information_schema.tables"

        sql_query=f"""
            SELECT COUNT(*)
            FROM {information_schema}
            WHERE table_name = '{tablename}'
            AND TABLE_SCHEMA = '{schema}'
            """

        out=pd.read_sql_query(sql_query, cnxn)
    except Exception as e:
        cnxn.rollback()
        sys.exit("Error in running SQL in MS Sql Server: \n" + str(e))
    
    if out.loc[0][0] == 1:
        cnxn.close()
        return True
    else:
        cnxn.close()
        return False

def last_date_MSql(db_name,
              date_col,
              config_file=config_file  
              ):
  
    if (MSql_table_check(db_name, config_file=config_file)): 
        sql_query=f"""select min({date_col}) as min_time ,
                             max({date_col}) as max_time
                        from   {db_name}"""
        saved_dates =mSql_query(sql_query, config_file=config_file) 
        last_saved_date=saved_dates['max_time'].iloc[0]
        print(f"The last date found in {db_name}:{last_saved_date}")
    else:
        print(f"{db_name} does not exist")
        last_saved_date=None

    return last_saved_date

def last_date_parquet(file_name, date_col):
    """
    Retrieves the most recent date from a specified date column in a Parquet file.

    Args:
        file_name (str): The path to the Parquet file.
        date_col (str): The name of the date column to check for the most recent date.

    Returns:
        datetime or None: The most recent date found in the specified date column, or None if the file does not exist.

    Prints:
        A message indicating the most recent date found in the file, or a message indicating that the file does not exist.
    """
    if (os.path.isfile(file_name)): 
        df= pd.read_parquet(file_name)
        last_saved_date=df[date_col].max()
        print(f"The last date found in {file_name}:{last_saved_date}")
    else:
        print(f"{file_name} does not exist")
        last_saved_date=None
    return last_saved_date

def last_date(output_list,
              **kwargs
              ):     
    import inspect 
    last_date_MSql_args = list(inspect.signature(last_date_MSql).parameters)
    kwargs_last_date_MSql= {k: kwargs.pop(k) for k in dict(kwargs) if k in last_date_MSql_args}
    date_col=output_list['date_col']
    if (output_list['format']=='MS_db') :
        last_save_date = last_date_MSql(output_list['output_location'],
                                    date_col,
                                    **kwargs_last_date_MSql
                                    )
    elif (output_list['format']=='parquet'):
        last_save_date = last_date_parquet(output_list['output_location'],
                            date_col,
                            )
    else:
        last_save_date=None
    
    return last_save_date

##TODO: merge the following code to last_date and remove it from spark_funcs
# def last_date(output_name,
#               date_col='date',
#               custom_config=None,
#               key_vault_dict='deltaTable',   ###for only delta_table
#               platform='databricks',         ### for only blob
#               ):
  
#   if (isinstance(output_name,str)) and (io_funcs.deltaTable_check(output_name)): 
#     saved_dates = io_funcs.query_deltaTable_db(
#                                               f"""select min({date_col}) as min_time ,
#                                                           max({date_col}) as max_time
#                                                   from   {output_name}""",
#                                               key_vault_dict=key_vault_dict,
#                                               custom_config=custom_config,
#                                               verbose=False 
#                                               ).toPandas()  
#     last_save_date=saved_dates['max_time'].iloc[0]
#     print(f"The last date found in delta_table:{last_save_date}")

#   elif (isinstance(output_name, dict)) and (io_funcs.blob_check(blob_dict=output_name,
#                                                                 custom_config=custom_config,
#                                                                 platform=platform,)):
#     udata = io_funcs.blob2pd(blob_dict=output_name,
#                             custom_config=custom_config,
#                             platform=platform,
#                             #  **kwargs_csv,
#                             )
#     udata[date_col] = pd.to_datetime(udata[date_col], format="%Y-%m-%d")
#     last_save_date= udata[date_col].max()  
    
#     print(f"The last date found in blob:{last_save_date}")

#   else:
#     last_save_date=None

#   return last_save_date

def update_output_specS(output_specS,
                        range_date__year=[2021,2099],
                        month_step=1,
                        firstDate=None,
                        lastDate=dt.datetime.now().date(),
                        ):
    """
    Updates the outputs list with the last saved date and generates a list of run dates.
    Parameters:
    output_specS (list or dict): A list or dictionary of outputs to be updated.
    range_date__year (list, optional): A list containing the start and end year for the date range. Default is [2021, 2099].
    month_step (int, optional): The step in months for generating the run dates. Default is 1.
    firstDate (str or pd.Timestamp, optional): The first date to start generating run dates. Can be a string in "YYYY-MM-DD" format or a pandas Timestamp. Default is None.
    lastDate (datetime.date, optional): The last date to generate run dates. Default is the current date.
    Returns:
    tuple: A tuple containing the updated outputs list and the list of run dates.
    Notes:
    - If `output_specS` is a dictionary, it will be converted to a list.
    - If `firstDate` is None, it will be set to the day after the last saved date if it exists.
    - A warning will be printed if `firstDate` is provided and the last saved date exists.
    - If no run dates are generated, a message indicating that the database or file is updated will be printed.
    - it uses only the first element of the output_specS to create the run_dates
    """

    output_specS= [output_specS] if isinstance(output_specS, dict) else output_specS
    output_specS2=output_specS
    last_saved_date__all=[]
    for con , output_list in enumerate(output_specS):
        # print(con, output_list)
        last_saved_date=last_date(output_list,
                                    config_file=config_file,
                                    )

        output_specS2[con]['last_date']=last_saved_date
        last_saved_date__all+=[last_saved_date]
        if con==0:
            warn_txt=False
            if (firstDate is not None):
                if isinstance(firstDate, str):
                    print(firstDate)
                    firstDate2   = dt.datetime.strptime(firstDate, "%Y-%m-%d").date()
            elif isinstance(firstDate,pd._libs.tslibs.timestamps.Timestamp):
                firstDate2   =firstDate.date()
                warn_txt=True  
            else:
                firstDate2= None if last_saved_date is None else last_saved_date+dt.timedelta(days=1)

            ###polish the warning, it is meaningless when last_saved_date exists
            if (warn_txt)& (last_saved_date is not None):
                print(f"The last date is {last_saved_date}; however, the function starts from given first date: {firstDate}")

            run_dates=cfuncs.datesList(range_date__year=range_date__year, 
                                    month_step=month_step,
                                    firstDate=firstDate2,
                                    lastDate=lastDate
                                    )

            if len(run_dates)==0:
                print("Database|file is updated")
            else:
                print("Date list updated to :\n", run_dates)

    if len(set(last_saved_date__all))>1:
        print("Warning! There are different last_date for output_specS, run_dates updated based on first element of output_specS:\t")
        print(last_saved_date__all)

    return output_specS2, run_dates

def save_outputs(output_dict, output_specS):
    """
    Save dataframes from the output dictionary to specified locations and formats.
    Parameters:
    output_dict (dict): A dictionary where keys are dataframe identifiers and values are the dataframes to be saved.
    output_specS (list or dict): A list of dictionaries or a single dictionary specifying the output details for each dataframe. 
                                 Each dictionary should contain the following keys:
                                 - 'output_df_key': The key in output_dict corresponding to the dataframe.
                                 - 'format': The format to save the dataframe ('MS_db' or 'parquet').
                                 - 'output_location': The location to save the dataframe.
                                 - 'overwrite': A boolean indicating whether to overwrite existing files.
    Returns:
    int: Returns 1 if the operation is successful, otherwise exits the program with an error message.
    Raises:
    SystemExit: If there are dataframes in output_dict that are not specified in output_specS.
    """
    flatten_ls=cfuncs.flattenList(output_dict['output_df_keys'])
    output_specS= [output_specS] if isinstance(output_specS, dict) else output_specS
    # output_specS2=[output_list for output_list in output_specS  if output_list['output_df_key'] in  flatten_ls]

    orphan_dfs=set(flatten_ls)-set([output_list['output_df_key'] for output_list in output_specS])
    if len(orphan_dfs)>0:
        print(f"Following dataframes are not saved: {orphan_dfs}")
        sys.exit("Please check the output_list and return values of the dfGenerator_func")
        return 0

    for key_dfS, df in zip(output_dict['output_df_keys'], output_dict['dfs']):
        if df.size!=0:
            for key_df in key_dfS:
                output_spec__sub=[output_list for output_list in output_specS  if output_list['output_df_key']==key_df][0]
                output_format=output_spec__sub['format']
                output_location=output_spec__sub['output_location']
                check_overwrite=output_spec__sub['overwrite']

                print(f"saving df in {output_format} format...")
                print(output_spec__sub)

                if (output_format=='MS_db'):  
                    
                    df2MSQL(df,
                            table_name=output_location.split('.')[2],
                            config_file=config_file,
                            schema=f"{output_location.split('.')[0]}.{output_location.split('.')[1]}",
                            chunksize=200,
                            method='multi',
                            index=False,
                            if_exists='replace' if check_overwrite else 'append'
                            )
                elif (output_format=='parquet'): 
                    if (not check_overwrite) and (os.path.isfile(output_location)):
                        df_current= pd.read_parquet(output_location)
                        # print(df.dtypes)
                        # print(df_current.dtypes)
                        df=pd.concat([df_current, df], axis=0)    
                    df.to_parquet(output_location, index=False)
                    
            print("Data saved successfully")
        else:
            print("Dataframe is empty")

    print('-'*200)
    return 1

##TODO: merge following codes to save_outputs and remove it from spark_funcs
# def save_outputs(ouputs_dict_list,
#                 **kwargs,
#               ):
#   import inspect
#   import dsToolbox.io_funcs as io_funcs
  
#   spark2del_args = list(inspect.signature(io_funcs.spark2deltaTable).parameters)
#   spark2del_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in spark2del_args}
#   pd2blob_args = list(inspect.signature(io_funcs.pd2blob).parameters)
#   pd2blob_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in pd2blob_args}

#   outputs=ouputs_dict_list.items() if isinstance(ouputs_dict_list, dict) else ouputs_dict_list

#   for (tableName_blobDict , sp) in outputs:
#     print(tableName_blobDict)
#     if isinstance(tableName_blobDict, str): 
#       io_funcs.spark2deltaTable(
#                                 sp,
#                                 table_name    = tableName_blobDict.split('.')[1],
#                                 schema        = tableName_blobDict.split('.')[0],
#                                 write_mode = 'append',
#                                 mergeSchema=True,
#                                 **spark2del_args
#                               )
#     elif isinstance(tableName_blobDict, dict): 
#       io_funcs.pd2blob(sp,
#                       blob_dict=tableName_blobDict,
#                       overwrite=False,
#                       append=True,
#                       **pd2blob_args
#                       )

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -----------------------Update a db table or a file recursively---------------------
##TODO: merge following codes to update_db_recursively and remove it from spark_funcs
def run_recursively(output_specS,
                    dfGenerator_func,
                    range_date__year=[2021,2099],
                    month_step=1,
                    firstDate=None,
                    lastDate=dt.datetime.now().date(),
                    **kwargs
                    ):
    """
    Executes a data generation function recursively over specified date ranges and saves the outputs.
    Parameters:
        output_specS (list)                : A list of dictionaries containing output configurations and update dates.
        dfGenerator_func (function)        : The data generation function to be executed.
        range_date__year (list, optional)  : A list containing the start and end years for the date range. Defaults to [2021, 2099].
        firstDate (datetime.date, optional): The start date for the date range. Defaults to None.
        lastDate (datetime.date, optional) : The end date for the date range. Defaults to the current date.
        **kwargs                           : Additional keyword arguments to be passed to the data generation function (dfGenerator_func).
    Returns:
        None
    Raises:
        Exception: If there is an error during the execution of the data generation function or saving the outputs.
    """
    # import src.io_funcs        as io_funcs
    import inspect
    print("updating the outputs list...")
    output_specS2, run_dates =update_output_specS(output_specS,
                                                range_date__year=range_date__year,
                                                month_step=month_step,
                                                firstDate=firstDate,
                                                lastDate=lastDate,
                                                )

    dfGenerator_func_args = list(inspect.signature(dfGenerator_func).parameters)
    dfGenerator_func_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in dfGenerator_func_args}

    print('-'*200)
    print('-'*200)
  
    try: 
        for ii in range(len(run_dates)-1):
            start_date, end_date= cfuncs.extract_start_end(run_dates, ii)
            print (f"Running {dfGenerator_func.__name__} for the period {start_date} to {end_date}...")
            output_dict=dfGenerator_func(start_date, end_date,
                                            **dfGenerator_func_args
                                            )
            save_outputs(output_dict, output_specS2)

    except Exception as e:
        print(f'***Running function {dfGenerator_func.__name__} failed: \n\t\t {str(e)}')
        print('**********************************************************************************************')