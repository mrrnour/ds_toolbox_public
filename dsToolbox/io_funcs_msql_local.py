import os,sys
import datetime as dt  
import pandas as pd
config_file=os.path.join('dsToolbox','config.yml')
import src.common_funcs    as cfuncs

class cred_setup_mssql():
    def __init__(self, config_file):
        import yaml
        from importlib import resources as res
        with open(config_file, 'r') as stream:
            self.config = yaml.safe_load(stream)

    def MSSQL_connector__pyodbc(self, db_server_id):
        import pyodbc 
        MSSQL_config=   self.config['sql_servers'][db_server_id]

        self.db_server = MSSQL_config['db_server']
        cnxn = pyodbc.connect("DRIVER={SQL Server Native Client 11.0};"
                            f"SERVER={self.db_server};"
                            "Trusted_Connection=yes;"
                            )
        print(f"Connected to {self.db_server}")
        return cnxn, MSSQL_config

    def MSSQL_connector__sqlalchemy(self, db_server_id):
        from sqlalchemy import create_engine
        import urllib
        MSSQL_config=   self.config['sql_servers'][db_server_id]

        self.db_server = MSSQL_config['db_server']
        # self.db_name   = MSSQL_config['db_name']

        db_params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+self.db_server+';Trusted_Connection=yes;')
        # db_params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+self.db_server+';DATABASE='+self.db_name+';Trusted_Connection=yes;')

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

def mSql_query(sql_query, db_server_id, config_file=config_file, return_df=True):
    import pandas as pd
    try:
        cnxn, _=cred_setup_mssql(config_file=config_file).MSSQL_connector__pyodbc(db_server_id)
        ##or using cursor:
        # cursor.execute(sql_query)
        # tables = cursor.fetchall()
        # for table in tables:
        #     print(table[0])

        # cnxn.execute(sql_query): This method is used to execute a SQL query against the database. The sql_query is a string that contains the SQL query you want to execute. This method does not return the results of the query. It is often used for SQL commands like INSERT, UPDATE, DELETE, etc., which modify the data but do not return any results.

        # cursor.fetchall(): This method is used to fetch all the rows of a query result. It returns a list of tuples where each tuple corresponds to a row in the result. You would typically use this method after executing a SELECT query to retrieve the data that the query returns.
    
        if return_df:
            # print(sql_query)
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

def df2MSQL(df, table_name, db_server_id, config_file=config_file, **kwargs):
    try:
        engine, _=cred_setup_mssql(config_file=config_file).MSSQL_connector__sqlalchemy(db_server_id)
        df.to_sql(table_name, con = engine,
                    **kwargs)

    except Exception as e:
        # engine.rollback()
        sys.exit("Error in writing dataFrame into MSSQL: \n" + str(e))

    return None

def table_exists(connection, table_name):
    """
    Check if a table exists in the database.
    
    Args:
        connection: psycopg2 database connection object
        table_name: name of the table to check (format: [database.]schema.table)
        
    Returns:
        bool: True if table exists, False otherwise
    """
    # Parse table name components
    table_parts = table_name.split(".")
    if len(table_parts) == 3:
        database, schema, table = table_parts
        information_schema = f"{database}.information_schema.tables"
    elif len(table_parts) == 2:
        schema, table = table_parts
        information_schema = "information_schema.tables"
    else:
        raise ValueError(f"Invalid table name format: {table_name}")
    
    with connection.cursor() as cursor:
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM {information_schema}
                WHERE table_schema = %s AND table_name = %s
            );
        """, (schema, table))
        return cursor.fetchone()[0]

##retire it after checking table_exists
def MSql_table_check(tablename,
                    db_server_id,
                    config_file=config_file,
                    ):
    import pandas as pd
    try:
        cnxn, _=cred_setup_mssql(config_file=config_file).MSSQL_connector__pyodbc(db_server_id)

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

def get_last_date_from_mssql_table(table_name, db_server_id, date_column, 
                                   config_file=config_file, logger=None):
    """
    Retrieve the most recent date from a specified column in an MS SQL table.
    
    Args:
        table_name (str): Name of the database table
        db_server_id (str): Database server identifier
        date_column (str): Name of the date column to query
        config_file (str): Configuration file path
        logger: Logger instance for custom logging
        
    Returns:
        datetime or None: The most recent date found, or None if table doesn't exist
    """
    
    # Check if table exists before querying
    if not MSql_table_check(table_name, db_server_id, config_file=config_file):
        cfuncs.custom_print(f"{table_name} does not exist", logger)
        return None
    
    # Query for min and max dates
    query = f"""
        SELECT MIN({date_column}) as min_time,
               MAX({date_column}) as max_time
        FROM {table_name}
    """
    
    date_results = mSql_query(query, db_server_id, config_file=config_file)
    most_recent_date = date_results['max_time'].iloc[0]
    
    print(f"The last date found in {table_name}: {most_recent_date}")
    
    return most_recent_date

##retire it after checking get_last_date_from_mssql_table
def last_date_MSql(db_name,
              db_server_id,
              date_col,
              config_file=config_file,
              logger=None  
              ):
  
    if (MSql_table_check(db_name, db_server_id, config_file=config_file)): 
        sql_query=f"""select min({date_col}) as min_time ,
                             max({date_col}) as max_time
                        from   {db_name}"""
        saved_dates =mSql_query(sql_query, db_server_id, config_file=config_file) 
        last_saved_date=saved_dates['max_time'].iloc[0]
        print(f"The last date found in {db_name}:{last_saved_date}")
    else:
        cfuncs.custom_print(f"{db_name} does not exist", logger)
        last_saved_date=None

    return last_saved_date

def last_date_parquet(file_name, date_col, logger=None):
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
        cfuncs.custom_print(f"The last date found in {file_name}:{last_saved_date}", logger)
    else:
        cfuncs.custom_print(f"{file_name} does not exist", logger)
        last_saved_date=None
    return last_saved_date

def last_date(output_dict,
              logger=None,
              **kwargs
              ):     
    import inspect 
    last_date_MSql_args = list(inspect.signature(last_date_MSql).parameters)
    kwargs_last_date_MSql= {k: kwargs.pop(k) for k in dict(kwargs) if k in last_date_MSql_args}
    date_col=output_dict['date_col']
    if (output_dict['format']=='MS_db') :
        db_server_id=output_dict['db_server_id']
        last_save_date = last_date_MSql(output_dict['output_location'],
                                    db_server_id,
                                    date_col,
                                    logger=logger,
                                    **kwargs_last_date_MSql
                                    )
    elif (output_dict['format']=='parquet'):
        last_save_date = last_date_parquet(output_dict['output_location'],
                            date_col,
                            logger=logger
                            )
    else:
        last_save_date=None
    
    return last_save_date

def load_parquet_between_dates(ufile   , 
                                date_col ,
                                start_date = '2019-01-01',
                                end_date   = '2020-01-01'):
    
    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end_date   = dt.datetime.strptime(end_date, "%Y-%m-%d")

    df= pd.read_parquet(ufile)
    # df['year']  = df[date_col].dt.year
    # df['month'] = df[date_col].dt.month
    # display(df.groupby(['year','month'])['SecureMessageThreadId'].count())

    df=df[(df[date_col]>=start_date)&(df[date_col]<end_date)]

    return df

def update_output_specS(output_specS,
                        range_date__year=[2021,2099],
                        month_step=1,
                        firstDate=None,
                        lastDate=dt.datetime.now().date(),
                        logger=None
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
    for con , output_dict in enumerate(output_specS):
        # print(con, output_dict)
        last_saved_date=last_date(output_dict,
                                    config_file=config_file,
                                    logger=logger
                                    )

        output_specS2[con]['last_date']=last_saved_date
        last_saved_date__all+=[last_saved_date]
        if con==0:
            warn_txt=False
            if (firstDate is not None):
                if isinstance(firstDate, str):
                    cfuncs.custom_print(firstDate, logger)
                    firstDate2   = dt.datetime.strptime(firstDate, "%Y-%m-%d").date()
            elif isinstance(firstDate,pd._libs.tslibs.timestamps.Timestamp):
                firstDate2   =firstDate.date()
                warn_txt=True  
            else:
                firstDate2= None if last_saved_date is None else last_saved_date+dt.timedelta(days=1)

            ###polish the warning, it is meaningless when last_saved_date exists
            if (warn_txt)& (last_saved_date is not None):
                cfuncs.custom_print(f"The last date is {last_saved_date}; however, the function starts from given first date: {firstDate}", logger)

            run_dates=cfuncs.datesList(range_date__year=range_date__year, 
                                    month_step=month_step,
                                    firstDate=firstDate2,
                                    lastDate=lastDate
                                    )

            if len(run_dates)==0:
                cfuncs.custom_print("Database|file is updated", logger)
            else:
                cfuncs.custom_print("Date list updated to :\n"+str(run_dates), logger)

    if len(set(last_saved_date__all))>1:
        cfuncs.custom_print("Warning! There are different last_date for output_specS, run_dates updated based on first element of output_specS:\t", logger)
        cfuncs.custom_print(last_saved_date__all, logger)

    return output_specS2, run_dates

def save_outputs(output_dict, output_specS, logger=None):
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
        cfuncs.custom_print(f"Following dataframes are not saved: {orphan_dfs}", logger)
        cfuncs.custom_print("Match output_list and return values of the dfGenerator_func",logger)
        sys.exit()
        # return 0

    orphan_outputs=set([output_list['output_df_key'] for output_list in output_specS])-set(flatten_ls)
    if len(orphan_outputs)>0:
        cfuncs.custom_print(f"Following outputs donot exist in dfGenerator_func: {orphan_outputs}", logger)
        cfuncs.custom_print("Match output_list and return values of the dfGenerator_func",logger)
        sys.exit()
        # return 0

    for key_dfS, df in zip(output_dict['output_df_keys'], output_dict['dfs']):
        if df.size!=0:
            for key_df in key_dfS:
                output_spec__sub=[output_list for output_list in output_specS  if output_list['output_df_key']==key_df][0]
                output_format=output_spec__sub['format']
                output_location=output_spec__sub['output_location']
                check_overwrite=output_spec__sub['overwrite']

                cfuncs.custom_print(f"saving output {output_spec__sub['output_df_key']} in {output_spec__sub['output_location']}...", logger)

                if (output_format=='MS_db'):  
                    db_server_id=output_spec__sub['db_server_id']
                    df2MSQL(df,
                            table_name=output_location.split('.')[2],
                            db_server_id=db_server_id,
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
                    
            # cfuncs.custom_print("Data saved successfully",logger)
        else:
            cfuncs.custom_print("Dataframe is empty", logger)

    cfuncs.custom_print('-'*50, logger)
    return 1



# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -----------------------Update a db table or a file recursively---------------------
def run_recursively(output_specS,
                    dfGenerator_func,
                    range_date__year=[2021,2099],
                    month_step=1,
                    firstDate=None,
                    lastDate=dt.datetime.now().date(),
                    logger=None,
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
    # import src.io_funcs_msql_local        as io_funcs
    import inspect
    cfuncs.custom_print("Updating the outputs list...\n", logger)
    output_specS2, run_dates =update_output_specS(output_specS,
                                                range_date__year=range_date__year,
                                                month_step=month_step,
                                                firstDate=firstDate,
                                                lastDate=lastDate,
                                                logger=logger
                                                )

    dfGenerator_func_args = list(inspect.signature(dfGenerator_func).parameters)
    dfGenerator_func_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in dfGenerator_func_args}

    cfuncs.custom_print('/'*50+'\n', logger)

    try: 
        for ii in range(len(run_dates)-1):
            start_date, end_date= cfuncs.extract_start_end(run_dates, ii)
            cfuncs.custom_print(f"Running {dfGenerator_func.__name__} for the period {start_date} to {end_date}...", logger)
            output_dict=dfGenerator_func(start_date, end_date,
                                         logger=logger,
                                            **dfGenerator_func_args
                                            )
            save_outputs(output_dict, output_specS2, logger=logger)

    except Exception as e:
        cfuncs.custom_print(f'***Running function {dfGenerator_func.__name__} failed: \n\t\t {str(e)}', logger)
        cfuncs.custom_print('************************************************************************', logger)
        cfuncs.custom_print(f'***Running function {dfGenerator_func.__name__} failed: \n\t\t {str(e)}', logger)
        cfuncs.custom_print('************************************************************************', logger)