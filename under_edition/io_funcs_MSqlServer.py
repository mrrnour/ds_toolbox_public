import os,sys

class cred_setup():
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

def mSql_query(sql_query, config_file=os.path.join('.','settings','config.yml'), return_df=True):
    import pandas as pd
    try:
        cnxn, _=cred_setup(config_file=config_file).MSSQL_connector__pyodbc()
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

def incorta_query(sql_query, config_file=os.path.join('.','settings','config.yml'), return_df=True):
    ###under edition: not tested yet
    import pandas as pd
    try:
        cnxn, _=cred_setup(config_file=config_file).incorta_connector()
        if return_df:
            out=pd.read_sql_query(sql_query, cnxn)
        else:
            out=cnxn.execute(sql_query)
    except Exception as e:
        cnxn.rollback()
        sys.exit("Error in running SQL in INCORTA: \n" + str(e))

    return out

def df2MSQL(df, table_name, config_file=os.path.join('.','settings','config.yml'), **kwargs):
    try:
        engine, _=cred_setup(config_file=config_file).MSSQL_connector__sqlalchemy()
        df.to_sql(table_name, con = engine,
                    **kwargs)

    except Exception as e:
        # engine.rollback()
        sys.exit("Error in writing dataFrame into MSSQL: \n" + str(e))

    return None
