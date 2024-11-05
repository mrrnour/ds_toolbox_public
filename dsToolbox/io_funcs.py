import os ,sys ,re, io
import datetime as dt

from importlib import resources as res
import pandas as pd
import numpy as np
import yaml

import dsToolbox.common_funcs   as cfuncs
import dsToolbox.default_values as par

__all__ = [ "query_synapse",
           'query_synapse_db',
           'query_synapse_local',
          "query_deltaTable",
          "query_template_run",
          
          "dbfs2blob",
          "spark2deltaTable",
          'deltaTable_check',
          
          "blob2spark",            
          "spark2blob",
          "blob2pd",
          "pd2blob",
          "pd2blob_batch",
          "blob_check",
          'xls2blob',
          
          "pi2pd_interpolate",
          "pi2pd_rawData",
          'pi2pd_seconds'
          ]

# upath="./dsToolbox/config.yml"
# upath='./sql_template.yml'
# config_yaml = yaml.safe_load(Path(upath).read_text())

def get_spark():
  import pyspark
  spark = pyspark.sql.SparkSession.builder.getOrCreate()
  sqlContext = pyspark.SQLContext(spark.sparkContext)
  return spark, sqlContext

def get_dbutils():
  import IPython
  dbutils = IPython.get_ipython().user_ns["dbutils"]
  return dbutils

def load_config(custom_config=None):
  if custom_config is None:
    with res.open_binary('dsToolbox', 'config.yml') as fp:
      config_yaml = yaml.load(fp, Loader=yaml.Loader)
  elif isinstance(custom_config, dict):  
    if ('key_vault_dictS' not in custom_config.keys())&\
      ('key_vault_name' in custom_config.keys())&\
      ('secret_name' in custom_config.keys())&\
      ('storage_account' in custom_config.keys())  :
      custom_config['key_vault_dictS']={}
      custom_config['key_vault_dictS'][custom_config['storage_account']]= {'key_vault_name' : custom_config['key_vault_name'],
                                                                          "secret_name"    : custom_config['secret_name']
                                                                          }
      for k in ('storage_account', 'key_vault_name', 'secret_name'):
          custom_config.pop(k)

    config_yaml =custom_config
  else:
    from pathlib import Path
    config_yaml = yaml.safe_load(Path(custom_config).read_text())

  key_vault_dictS      = config_yaml.get('key_vault_dictS')
  KV_access_local      = config_yaml.get('KV_access_local')
  synapse_cred_dict    = config_yaml.get('synapse_cred_dict')
  azure_ml_appID       = config_yaml.get('azure_ml_appID')
  pi_server_dict       = config_yaml.get('pi_server')
  # print(key_vault_dictS, KV_access_local, synapse_cred_dict, azure_ml_appID)
  return config_yaml, key_vault_dictS, KV_access_local, synapse_cred_dict, azure_ml_appID, pi_server_dict

io_config_dict, _, _, _, _, _=load_config(custom_config=None)

class cred_strings():
  def __init__(self, key_vault_dict, custom_config=None, platform='databricks'):
    
    _, key_vault_dictS, KV_access_local, synapse_cred_dict, azure_ml_appID, pi_server_dict=load_config(custom_config)
    
    self.key_vault_dict  = key_vault_dict
    cred_dict            = key_vault_dictS[key_vault_dict]
    self.key_vault_name  = cred_dict.get('key_vault_name')
    self.secret_name     = cred_dict.get('secret_name')

    self.platform        = platform
    self.azure_ml_appID  = azure_ml_appID
    self.KV_access_local = KV_access_local
    
    self.synapse_cred_dict = synapse_cred_dict
    
    self.pi_server_dict  = pi_server_dict
    
    self.password        = fetch_key_value(self.key_vault_name,
                                            self.secret_name,
                                            self.azure_ml_appID,
                                            self.KV_access_local,
                                            self.platform)

  def blob_connector(self, filename, container):
    self.storage_account = self.key_vault_dict
    blob_host= f"fs.azure.account.key.{self.storage_account}.blob.core.windows.net"

    path = f'{container}@{self.storage_account}'
    blob_path = f"wasbs://{path}.blob.core.windows.net/{filename}"
    blob_connectionStr= (f'DefaultEndpointsProtocol=https;AccountName={self.storage_account};'
                          f'AccountKey={self.password};EndpointSuffix=core.windows.net')
    
    return blob_host, blob_path, blob_connectionStr
    
  def spark_host(self):
    return f"fs.azure.account.key.{self.key_vault_name}.dfs.core.windows.net"

  def synapse_connector(self):
    hostname = self.synapse_cred_dict['hostname']
    database = self.synapse_cred_dict['database']
    port     = self.synapse_cred_dict['port']
    username = self.synapse_cred_dict['username']
    driver   = self.synapse_cred_dict.get('driver')
    driver_odbc=self.synapse_cred_dict.get('driver_odbc')
    port       =self.synapse_cred_dict['port']
    
    properties = {"user"     : username,
                  "password" : self.password,
                  "driver"   : driver }

    url = f"jdbc:sqlserver://{hostname}:{port};database={database}"

    odbc_connector = (f"DRIVER={driver_odbc};SERVER={hostname};PORT={port};"
                      f"DATABASE={database};UID={username};"
                      f"PWD={self.password}; MARS_Connection=yes")
        
    return url, properties, odbc_connector
    
  #Call to Azure to retrieve OAuth access token (required for all PI API calls through cloud services)
  def pi_server_connector(self):
    url = self.pi_server_dict['url']
    myobj = {'grant_type'  : self.pi_server_dict['grant_type'],
            'client_id'    : self.pi_server_dict['client_id'],
            'scope'        : self.pi_server_dict['client_secret'],
            'client_secret': self.password
            }
    import requests
    oAuthResponse = requests.post(url, data = myobj)
    accessToken = oAuthResponse.json().get('access_token')
    return accessToken

def clean_query(q,
                # n=100
                ):
    q = q.strip().lstrip('(')
    q = q.rstrip('query')
    q = q.strip().rstrip(')')
    # q = q.replace('SELECT', f'SELECT TOP({n})')
    return q
  
def get_secret_KVUri(key_vault_name, secret_name, credential):
  from azure.keyvault.secrets import SecretClient    
  KVUri      = f"https://{key_vault_name}.vault.azure.net"
  client     = SecretClient(vault_url = KVUri, credential = credential)
  secret     = client.get_secret(secret_name).value
  return secret

def fetch_key_value(key_vault_name, secret_name,
                     azure_ml_appID, KV_access_local,
                     platform='databricks',):
  if platform=='databricks':
    # print('i am databricks run')
    dbutils = get_dbutils()
    secret = dbutils.secrets.get(scope =key_vault_name, key = secret_name)

  elif platform == 'aml':
    print("using azure ML and managed identity authentication")
    #Auth Mechanism: Managed Identity
    if azure_ml_appID is None:
      sys.exit("Identity Application ID is not provided")
    from azure.identity import ManagedIdentityCredential
    client_id                  = f"{azure_ml_appID}"  ##EnterManagedIdentityApplicationID
    credential                 = ManagedIdentityCredential(client_id = client_id)
    credential.get_token("https://vault.azure.net/.default")
    secret=get_secret_KVUri(key_vault_name, secret_name, credential = credential)

  elif platform == 'local' or 'vm_docker':
    print('i am locally run')

    from azure.identity import DefaultAzureCredential    
    try:    
      import os
      if os.environ.get('AZURE_TENANT_ID') is None:      
        os.environ['AZURE_TENANT_ID']     = KV_access_local['secret_TenantID']
      if os.environ.get('AZURE_CLIENT_ID') is None:    
        os.environ['AZURE_CLIENT_ID']     = KV_access_local['secret_ClientID__prd']
      if os.environ.get('AZURE_CLIENT_SECRET') is None:      
        os.environ['AZURE_CLIENT_SECRET'] = KV_access_local['secret_ClientSecret__Prd']
    except Exception as e:
      print(f'{str(e)}')

      if KV_access_local is None:
        sys.exit("""set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET environment variables 
                    or provide KV_access_local dictionary in config.yml file to extract them """)

    from azure.identity import DefaultAzureCredential    
    secret=get_secret_KVUri(key_vault_name, secret_name, credential = DefaultAzureCredential())

  return secret

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ----------------------------Running queries -----------------------------
def query_synapse(query: str,
                  platform='databricks',
                  key_vault_dict: str ='azure_synapse',
                  custom_config=None,
                  verbose=True):
  """Run a Query in azure synapse

  Params:
    
    query: (string) SQL query string
    
    verbose:(Boolean)  an option for producing detailed information
    
  Returns: a spark dataframe in databricks or pandas dataframe in local/ vm_docker
  """
  
  if platform=='databricks':
    query_synapse_db(query,
                    key_vault_dict=key_vault_dict,
                    custom_config=custom_config,
                    verbose=verbose,
                    )
  elif (platform=='local') or (platform=='vm_docker'):
    query_synapse_local(query,
                        key_vault_dict=key_vault_dict,
                        custom_config=custom_config,
                        verbose=verbose,
                        )

def query_synapse_db(query: str,
                    key_vault_dict: str ='azure_synapse',
                    custom_config=None,
                    verbose=True,
                    ):
  """Run a Query in azure synapse
  
    Params:
     query: (string) SQL query string
      
      key_vault_dict(string) dictionary name in config.yml
      
      verbose:(Boolean)  an option for producing detailed information
      
      custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead
      
    Returns: a spark dataframe
  """
  c=cred_strings(key_vault_dict=key_vault_dict,
                 custom_config=custom_config,
                 platform='databricks')
  url, properties, _=c.synapse_connector()

  query=f'({query}) query' if (query.strip()[-5:]!='query') or (query.strip()[0]!='(') else query 
  if verbose: print("pulling data from azure_synapse:\n", query) 

  spark, sqlContext=get_spark()
  df  = spark.read.jdbc(table=query, url=url, properties=properties)

  ###----for local:
  # query = clean_query(query)  
  # import pyodbc
  # con = pyodbc.connect(cnstr)
  # df  = pd.read_sql(query,con)
  # con.close()

  return df  

def query_synapse_local(query: str,
                        key_vault_dict: str ='azure_synapse',
                        custom_config=None,
                        verbose=True,
                        ):
  """Run a Query in azure synapse
  
    Params:
    
      query: (string) SQL query string
      
      key_vault_dict(string) dictionary name in config.yml
      
      custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead

      verbose:(Boolean)  an option for producing detailed information
      
    Returns: a pandas dataframe
  """
  # query=query.strip()
  # if (query[-5:]=='query'):
  #   query=query[1:-5]

  c=cred_strings(key_vault_dict=key_vault_dict,
                 custom_config=custom_config,
                 platform='local')
  _, _, cnstr=c.synapse_connector()

  query = clean_query(query)  
  if verbose: print("pulling data from azure_synapse:\n", query) 

  import pyodbc
  con = pyodbc.connect(cnstr)
  df  = pd.read_sql(query,con)
  con.close()
  return df

def query_deltaTable_db(query: str,
                        key_vault_dict: str ='deltaTable',
                        verbose=True,
                        custom_config=None,
                        ):
  """Run a Query in deltaTable
  
    Params:
     query: (string) SQL query string
      
      key_vault_dict(string) dictionary name in config.yml
      
      verbose:(Boolean)  an option for producing detailed information
      
      custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead
      
    Returns: a spark dataframe 
  """

  if verbose: print("pulling data from deltaTable:\n", query) 
  c = cred_strings(key_vault_dict=key_vault_dict,
                   custom_config=custom_config,
                   platform='databricks')
  spark, sqlContext=get_spark()
  spark.conf.set(c.spark_host(), c.password)
  df = spark.sql(query)
  return df

def query_template_reader(query_str: str,
                          replace_dict: dict={'start___date':par.start_date,
                                              'end___date'  :par.end_date
                                              },
                          ):
    ##TODO: check required format based on yaml 
    if cfuncs.check_timestamps(replace_dict.get('start___date'), replace_dict.get('end___date')):
      for key,value in replace_dict.items():
        query=query_str.replace(key,value)  
    return query

def query_template_run(query_temp_name: str,
                      replace_dict: dict={'start___date':par.start_date,
                                          'end___date'  :par.end_date
                                          },
                      custom_config=None,
                      custom_sql_template_yml=None,
                      platform='databricks',
                      ):
  """Run a Query in deltaTable or azure synapse
    Params:
      query_temp_name: (dictionary) the name of dictionary that contains query template
      
      custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead
      
      custom_sql_template_yml(path): a path of predefined  SQL qureies, if it is not provided, dsToolbox.sql_template_dict will be used instead
      
      replace_dict:(dictionary) it is used to replace start and end date
      
    Returns: a spark dataframe
  """
  
  if custom_sql_template_yml is None:
    with res.open_binary('dsToolbox', 'sql_template.yml') as fp:
      sql_template_dict = yaml.load(fp, Loader=yaml.Loader)
  else:
    from pathlib import Path
    sql_template_dict = yaml.safe_load(Path(custom_sql_template_yml).read_text())

  tmp=sql_template_dict.get(query_temp_name)
  host, query_str = tmp['db'], tmp['query'] 

  query=query_template_reader(query_str,
                            replace_dict=replace_dict
                            )
  if host=='azure_synapse':
    if platform=='databricks':
      df=query_synapse_db(
                          query,
                          key_vault_dict=host,
                          custom_config=custom_config,
                          verbose=True,
                          )
    elif (platform=='local') or (platform=='vm_docker'):
      df=query_synapse_local(query,
                            key_vault_dict=host,
                            custom_config=custom_config,
                            verbose=True,
                            )
  else:
    df=query_deltaTable_db(
                          query,
                          key_vault_dict=host,
                          custom_config=custom_config,
                          )  
  return df

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ----------------------------Talking with databricks  ------------------
def dbfs2blob(ufile, blob_dict, custom_config=None):      
  """Save a File from databricks into Azure blob storage
    Params:
      ufile  : (string) Path to file saved on Databricks File System (DBFS) starting with '/dbfs/'
      
      blob_dict:(Dictionary) TargetFile path   : {'key_vault_dict':storage account_name(string) ,
                                                  'container':container_name(string) ,
                                                  'blob': blob_name(string) 
                                                  }     
                                                    
    Returns: None
  """                       
  storage_account = blob_dict.get('storage_account')
  container       = blob_dict.get('container')  
  blob            = blob_dict.get('blob')

  c = cred_strings(key_vault_dict=storage_account,
                   custom_config=custom_config,
                   platform='databricks')
  blob_host, blob_path, blob_connectionStr=c.blob_connector(blob, container)

  ##TODO: setting spark with blob_host?
  spark, sqlContext=get_spark()
  spark.conf.set(blob_host, c.password)
  
  dbutils=get_dbutils()
  dbutils.fs.cp(ufile.replace("/dbfs","dbfs:"),blob_path)
  print(f"{ufile} saved in {blob_path}")

def spark2deltaTable(df, table_name: str, schema: str = 'xxx_analytics',
                    write_mode:str = 'append', partitionby:list = None, 
                    **options
                    ):
  """ Writes to databricks delta tables
  
    Params: 
          df           : Spark Dataframe to write to delta table
            
          database     : (string) Name of database
            
          table        : (string) Name of table. Creates a new table if it doesn't exist
            
          write_mode   : (string) Write mode ('append' or 'overwrite'). Default is 'append'
            
          partitionby  : (list) list of Column names to partition by.

          **options    : (dict) all other string options 
            
    Return: None
  """
  spark, sqlContext=get_spark()
  spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema}")

  if partitionby is None:
    partitionBy=''
  else:
    partitionby=[partitionby] if not isinstance(partitionby, list) else partitionby
    partitionBy=f', partitionby={partitionby}'

  eval(f"df.write.saveAsTable('{schema}.{table_name}', mode='{write_mode}' {partitionBy} , **{options})")
  
  return

def deltaTable_check(delta_tableName: str,
                     ) -> bool:

  """check a delta table exist or not
  
    Params:
    
    delta_tableName:(string) the tablename

    Returns: (Boolean)
  """  
  spark, sqlContext=get_spark()
  is_delta = spark._jsparkSession.catalog().tableExists(delta_tableName)
  ###https://kb.databricks.com/en_US/delta/programmatically-determine-if-a-table-is-a-delta-table-or-not
  # desc_table = spark.sql(f"describe formatted {delta_tableName}").collect()
  # location = [i[1] for i in desc_table if i[0] == 'Location'][0]
  # try:
  #   dir_check = dbutils.fs.ls(f"{location}/_delta_log")
  #   is_delta = True
  # except Exception as e:
  #   is_delta = False

  if is_delta:
    print(f"table {delta_tableName} exists!")
  return is_delta

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ----------------------------Talking with Azure Storage ------------------
##storage account-->containers-->blob
def blob2spark(blob_dict:dict,
              custom_config=None,
              platform='databricks'
              ):
  """read a blob file (csv or parquet file) as a spark dataframe
    Params:
    
      blob_dict:(Dictionary) TargetFile path   : {'key_vault_dict':storage account_name(string) ,
                                                  'container':container_name(string) ,
                                                  'blob': blob_name(string) 
                                                  }  
                                                        
      verbose:(Boolean)  an option for producing detailed information   
      
      custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead
                            
    Returns: spark dataframe
  """  
  storage_account=blob_dict.get('storage_account')
  container=blob_dict.get('container')  
  blob=blob_dict.get('blob')
  
  c = cred_strings(key_vault_dict=storage_account,
                   custom_config=custom_config,
                   platform=platform)
  blob_host, blob_path, blob_connectionStr=c.blob_connector(blob, container)
  spark, sqlContext=get_spark()
  spark.conf.set(blob_host, c.password)
  extension=blob.split('.')[-1]
  if extension=='csv':
    df = spark.read.format('csv')\
                  .option('header','true')\
                  .option('inferSchema','true')\
                  .load(blob_path)
  elif extension=='parquet':
    df= spark.read.format("parquet")\
                  .load(blob_path)
  return df

def spark2blob(df,
              blob_dict:dict,
              write_mode:str = "mode('append')",
              custom_config=None,
              platform='databricks'
              ):
  """Save spark dataframe (df) into Azure blob storage using df.write.format command
    Params:
    
    df: park dataframe
    
    blob_dict:(Dictionary) TargetFile path   : {'key_vault_dict':storage account_name(string) ,
                                                  'container':container_name(string) ,
                                                  'blob': blob_name(string) 
                                                  }  

    write_mode(string)  attached like options to df.write.format command   
    
    custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead

    Returns: None
  """          
  storage_account = blob_dict.get('storage_account')
  container       = blob_dict.get('container')  
  blob            = blob_dict.get('blob')

  c = cred_strings(key_vault_dict=storage_account,
                   custom_config=custom_config,
                   platform=platform)
  blob_host, blob_path, blob_connectionStr=c.blob_connector(blob, container)
  spark, sqlContext=get_spark()
  spark.conf.set(blob_host, c.password)
  extension=blob.split('.')[-1]
  
  string_run=f"""df.write.format('{extension}')\
                .{write_mode}
              """

  string_run=f"""{string_run}
                  .save(blob_path)"""  
  string_run=re.sub(r"[\n\t\s]*", "", string_run)
  print("running:\n", string_run)
  eval(string_run)

def blob2pd(blob_dict:dict,
            verbose=True,
            custom_config=None,
            platform='databricks',
            load_to_memory=False,
            **kwargs
            )-> pd.DataFrame:
  """read a blob file (csv or parquet file) as a panadas dataframe
  
    Params:
    
    blob_dict:(Dictionary) TargetFile path   : {'key_vault_dict':storage account_name(string) ,
                                                  'container':container_name(string) ,
                                                  'blob': blob_name(string) 
                                                  }

    verbose:(Boolean)  an option for print detailed information
    
    custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead
    
    load_to_memory:(Boolean)  This option allows loading directly into memory rather than using temporary files

    **kwargs: options used in pd.read_csv or pd.read_parquet   

    Returns: panda dataframe
  """   
  import inspect 
  csv_args = list(inspect.signature(pd.read_csv).parameters)
  kwargs_csv = {k: kwargs.pop(k) for k in dict(kwargs) if k in csv_args}
  # print("csv arugments:", kwargs_csv)
  parq_args = list(inspect.signature(pd.read_parquet).parameters)
  kwargs_parq = {k: kwargs.pop(k) for k in dict(kwargs) if k in parq_args}
  # print(f"parquet arugments:", parq_args)

  from azure.storage.blob import BlobServiceClient
  # from azure.storage.blob import ContainerClient

  storage_account = blob_dict.get('storage_account')
  container       = blob_dict.get('container')  
  blob            = blob_dict.get('blob')

  c = cred_strings(key_vault_dict=storage_account,
                   custom_config=custom_config,
                   platform=platform)
  blob_host, blob_path, blob_connectionStr=c.blob_connector(blob, container)
  blob_service_client = BlobServiceClient.from_connection_string(blob_connectionStr)
  blob_client         = blob_service_client.get_blob_client(container = container, blob = blob)
  extension           = blob.split('.')[-1]

  if verbose: print(f"Downloading from storage_account:'{storage_account}', container:'{container},' and blob:'{blob}' as bytes")
  
  if load_to_memory:
    with io.BytesIO() as blob_dest:
      blob_client.download_blob().readinto(blob_dest)
      blob_dest.seek(0)
      if extension=='csv':
        df = pd.read_csv(blob_dest,**kwargs_csv)
      elif extension=='parquet':
        df = pd.read_parquet(blob_dest,**kwargs_parq)  
      else:
        print(f"file uploaded into the memory")
        df=blob_dest
    
  else:  
    import os
    blob_file=os.path.basename(blob)
    tmp_file_locs={
                  'databricks' : f'/tmp/{blob_file}',
                  'aml'        : f'{os.getcwd()}/{blob_file}',
                  'local'      : f'{os.getcwd()}/{blob_file}',
                  'vm_docker'  : f'{os.getcwd()}/{blob_file}',
                  }
  
    # if platform=='databricks':
    #   dbutils=get_dbutils()
    #   dbutils.fs.mkdirs("/tmp/")

    tmp_file=tmp_file_locs[platform]
    with open(tmp_file, 'wb') as blob_dest:
      data = blob_client.download_blob()
      blob_dest.write(data.readall())
    if extension=='csv':
      df = pd.read_csv(tmp_file,**kwargs_csv)
      os.remove(tmp_file)
    elif extension=='parquet':
      df = pd.read_parquet(tmp_file,**kwargs_parq)  
      os.remove(tmp_file)
    else:
      print(f"file uploaded in {tmp_file}")
      df=tmp_file
  
  return df

def pd2blob(data: pd.DataFrame,
            blob_dict:dict,
            append=False,
            overwrite=True,
            platform='databricks',
            custom_config=None,
            sheetName='dataframe1',
            **kwargs
          ):
  """Save pandas dataframe (df) into Azure blob storage using blob_client
  
    Params:
    
    data: pandas dataframe
    
    blob_dict:(Dictionary) TargetFile path   : {'key_vault_dict':storage account_name(string) ,
                                                  'container':container_name(string) ,
                                                  'blob': blob_name(string) 
                                                  }  

    append(boolean):       when append=True, append dataframe to existing file 
    
    sheetName(string):     sheet name in the excel file
    
    overwrite(boolean):    when overwrite=True, overwrite to  existing file  

    custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead
    

    **kwargs: args for :
      blob_client.upload_blob see:https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blobclient?view=azure-python#azure-storage-blob-blobclient-upload-blob
      pandas.DataFrame.to_csv
      pandas.DataFrame.to_parquet
      pandas.DataFrame.to_excel
                    
    Returns: get_blob_properties
  """      
  ##for more information please see 
  ##https://docs.microsoft.com/en-us/python/api/overview/azure/storage-blob-readme?view=azure-python#key-concepts
  ##https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction
  ##https://github.com/Azure/azure-sdk-for-python/tree/azure-storage-blob_12.9.0/sdk/storage/azure-storage-blob/samples

  import sys, io
  import pandas as pd
  ###using BlobServiceClient    
  from azure.storage.blob import BlobServiceClient
  storage_account = blob_dict.get('storage_account')
  container       = blob_dict.get('container')  
  blob            = blob_dict.get('blob')

  c = cred_strings(key_vault_dict=storage_account,
                   custom_config=custom_config,
                   platform=platform)
  blob_host, blob_path, blob_connectionStr=c.blob_connector(blob, container)
  blob_service_client = BlobServiceClient.from_connection_string(blob_connectionStr)
  container_client    = blob_service_client.get_container_client(container)
  
  ##method1:
  # blob_client = container_client.upload_blob(name=blob,
  #                                            data=data,
  #                                            **kwargs
  #                                           )

  ##method2:
  blob_client=container_client.get_blob_client(blob)    

  if blob_client.exists():
    print(f"File Exists!")
    if  (append==overwrite) :             ###('overwrite' in kwargs)and (kwargs.get("overwrite")==append):
      print(f"append and overwrite have values same = {append}",'\n Append set to True and overwrite set to False')    
      append=True
      overwrite=False

  ##method3:
  ##https://medium.com/featurepreneur/parquet-on-azure-27725ab1246b
  #blob_client = blob_service_client.get_blob_client(container = container, blob = blob)

  # print(f"Uploading file: {} to key_vault_dict:'{key_vault_dict}' container:'{container},' and blob:'{blob}' as bytes")
  import inspect
  
  csv_args = list(inspect.signature(pd.DataFrame.to_csv).parameters)
  kwargs_csv = {k: kwargs.pop(k) for k in dict(kwargs) if k in csv_args}

  parq_args = list(inspect.signature(pd.DataFrame.to_parquet).parameters)
  kwargs_parq = {k: kwargs.pop(k) for k in dict(kwargs) if k in parq_args}

  xls_args = list(inspect.signature(pd.DataFrame.to_excel).parameters)
  kwargs_xls = {k: kwargs.pop(k) for k in dict(kwargs) if k in xls_args}

  blob_args = list(inspect.signature(blob_client.upload_blob).parameters)
  kwargs_blob = {k: kwargs.pop(k) for k in dict(kwargs) if k in blob_args}
  
  if blob.split('.')[1] == 'csv':
    if blob_client.exists() and append:
      blob_client.upload_blob(data=data.to_csv(header=False, **kwargs_csv),
                              **kwargs_blob,
                              blob_type="AppendBlob"
                              )
    else:
      blob_client.upload_blob(data=data.to_csv(**kwargs_csv),
                              **kwargs_blob, overwrite=overwrite,
                              blob_type="AppendBlob"
                              )
  elif blob.split('.')[1] == 'parquet':
    if blob_client.exists() and append:
      df_current = blob2pd(blob_dict)
      df_current = pd.concat([df_current, data],axis=0)
      blob_client.upload_blob(data=df_current.to_parquet(**kwargs_parq),
                              overwrite=True,
                              **kwargs_blob)
    else:
      blob_client.upload_blob(data=data.to_parquet(**kwargs_parq),
                              overwrite=overwrite,
                              **kwargs_blob, 
                              ) 

  elif blob.split('.')[1]=='xls' :
    sys.exit("the function does not support the old .xls file format, please use xlsx format")
 
  elif blob.split('.')[1]=='xlsx' :
    import openpyxl, io
    if (append) &(~overwrite):
      sys.exit("the function does not append to an existing excel file, use xls2blob to save multiple dataframes to a excel file")    
    else:
      output = io.BytesIO()
      with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
          data.to_excel(writer, sheet_name=sheetName, **kwargs_xls)
      xlsx_data = output.getvalue() 
      blob_client.upload_blob(data=xlsx_data,
                                overwrite=True,
                                **kwargs_blob,
                                )
  else:
      print("Append option is not usable")
      blob_client.upload_blob(data=data,
                              overwrite=overwrite,
                              **kwargs_blob,
                              )        
  return blob_client.get_blob_properties()

def pd2blob_batch(outputs:dict,
                  blob_dict={'container':'xxx', 
                              'key_vault_dict':'prdadlafblockmodel'},
                  append=True ,
                  platform='databricks',
                  **kwargs):
  """Save pandas dataframes (df) into Azure blob storage using df.write.format command
      Params:
          
      outputs: (Dictionary)  key: blob path   , value:dataframe
      
      blob_dict:(Dictionary) TargetFile path   : {'key_vault_dict':storage account_name(string) ,
                                                  'container':container_name(string) ,
                                                  'blob': blob_name(string) 
                                                  }  

      append(boolean):      when append=True, append dataframe to existing file   

      write_mode(string)  attached like options to df.write.format command    

      **kwargs: args for blob_client.upload_blob see:https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blobclient?view=azure-python#azure-storage-blob-blobclient-upload-blob
  """    
  for out in outputs:
    try:
      blob_dict['blob']=out
      pd2blob(outputs.get(out),
              blob_dict=blob_dict,
              platform=platform,
              append=append,
              **kwargs
              )
      print(f'{out} saved')
    except Exception as e:
      print(f'***writing {out} failed: \n\t\t {str(e)}')
      pass

def blob_check(blob_dict:dict,
               custom_config=None,
               platform='databricks'):
  """check a blob exist or not
  
    Params:
    
    blob_dict:(Dictionary) file path  : {'key_vault_dict':storage account_name(string) ,
                                            'container':container_name(string) ,
                                            'blob': blob_name(string) 
                                        } 

    Returns: (Boolean)
  """  
  from azure.storage.blob import BlobClient
  
  storage_account = blob_dict.get('storage_account')
  container       = blob_dict.get('container')  
  blob            = blob_dict.get('blob')

  c = cred_strings(key_vault_dict=storage_account,
                   custom_config=custom_config,
                   platform=platform)
  blob_host, blob_path, blob_connectionStr=c.blob_connector(blob, container)

  blob = BlobClient.from_connection_string(conn_str=blob_connectionStr,
                                           container_name=container,
                                           blob_name=blob)

  return blob.exists()

def xls2blob(dataframe_dict: dict,
            blob_dict:dict,
            overwrite=True,
            custom_config=None,
            platform='databricks',
            **kwargs
          ):
  """Save pandas dataframe(s) as a excel file into Azure blob storage
    Params:
    
    dataframe_dict:  dictionary of sheet_name annd corresponding dataframes to write
    
    blob_dict:(Dictionary) TargetFile path   : {'key_vault_dict':storage account_name(string) ,
                                                  'container':container_name(string) ,
                                                  'blob': blob_name(string) 
                                                  }  
    
    overwrite(boolean):    when overwrite=True, overwrite to  existing file  

    custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead

    **kwargs: args for :
      blob_client.upload_blob see:https://docs.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.blobclient?view=azure-python#azure-storage-blob-blobclient-upload-blob
      pandas.DataFrame.to_excel
                    
    Returns: get_blob_properties
  """      
  ###using BlobServiceClient    
  from azure.storage.blob import BlobServiceClient
  storage_account = blob_dict.get('storage_account')
  container       = blob_dict.get('container')  
  blob            = blob_dict.get('blob')

  c = cred_strings(key_vault_dict=storage_account,
                   custom_config=custom_config,
                   platform=platform)
  blob_host, blob_path, blob_connectionStr=c.blob_connector(blob, container)
  blob_service_client = BlobServiceClient.from_connection_string(blob_connectionStr)
  container_client    = blob_service_client.get_container_client(container)
  
  blob_client=container_client.get_blob_client(blob)    

  import inspect
  import sys, io
  import pandas as pd
    
  blob_args = list(inspect.signature(blob_client.upload_blob).parameters)
  kwargs_blob = {k: kwargs.pop(k) for k in dict(kwargs) if k in blob_args}
  
  xls_args = list(inspect.signature(pd.DataFrame.to_excel).parameters)
  kwargs_xls = {k: kwargs.pop(k) for k in dict(kwargs) if k in xls_args}

  output = io.BytesIO()
  writer =pd.ExcelWriter(output, engine='xlsxwriter')
  for sheetName, df in  dataframe_dict.items(): 
    df.to_excel(writer, sheet_name=sheetName, **kwargs_xls)
        
  writer.close()
  xlsx_data = output.getvalue() 
  blob_client.upload_blob(data=xlsx_data,
                          overwrite=overwrite,
                          **kwargs_blob,
                          )
  return blob_client.get_blob_properties()

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
#Call  PI Cloud API's point controller to get WebID for tag list in MST (WebID = persistent unique ID for a PI tag, required for majority of PI api calls)
def get_web_ids(accessToken, tags):
  import requests, json
  web_ids = {}
  for tag in tags:
    url = 'https://svc.apiproxy.exxonmobil.com/KRLPIV01/v1/piwebapi/points?path=\\KRLPIH01\{}'.format(tag)
    headers = {
      'Authorization':'Bearer ' + accessToken,
      'Content-Type':'application/json'}
    response = requests.request("GET", url, headers=headers)

    web_id=json.loads(response.text).get('WebId')
    web_ids[tag] =web_id

    if web_id is None:
      print(f"PI tag not found: {tag}")
  return web_ids

def pi2pd_interpolate(tags,
                    start_date=par.start_date, end_date=par.end_date,
                    interval = '1h',
                    pi_vault_dict='webapi', 
                    custom_config=None,
                    platform='databricks'):
  """
  Call PI Cloud API's stream controller to get interpolated data for tags  
  Get pi tag data according to desired frequency
  
  Params:
  
  tags(list)                  : maximum 11 tags at a time
    
  start_date (string)          : start date, format: "%Y-%m-%d"
  
  end_date   (string)          : end date, format: "%Y-%m-%d"
  
  pi_vault_dict(String)   : key_vault_dict dictionary  name for webapi in azure_valuet_cred dictionary     
  
  interval(string)  '1h'      : get interpolated data (default hourly)
  
  custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead
  
  """ 
  import requests, urllib, json
  if isinstance(start_date,str):
    start_date   = dt.datetime.strptime(start_date, "%Y-%m-%d") ###'2021-12-31', "%Y-%m-%d"
  
  if isinstance(end_date, str):
    end_date     = dt.datetime.strptime(end_date,   "%Y-%m-%d") ###'2021-12-31', "%Y-%m-%d"

  if not isinstance(tags, list):tags=[tags]

  c = cred_strings(key_vault_dict=pi_vault_dict,
                   custom_config=custom_config,
                   platform=platform)
  accessToken = c.pi_server_connector()
  web_ids     = get_web_ids(accessToken, tags)
  tagData     = {}
  for i, tag in enumerate(tags):
    webID = web_ids[tag]
    print('tag=',tag,",webID=",f"{webID[:5]}...{webID[20:25]}...{webID[-5:]}")
    if webID is not None:
      query_dict = {'startTime':start_date, 'endTime':end_date, 'interval':interval}
      url = 'https://svc.apiproxy.exxonmobil.com/KRLPIV01/v1/piwebapi/streams/'+webID+'/interpolated?'+urllib.parse.urlencode(query_dict)
      headers = {
        'Authorization':'Bearer ' + accessToken,
        'Content-Type':'application/json'}
      response = requests.request("GET", url, headers=headers)
      json_str = json.loads(response.text)
      if i == 0:
          tagData['Date'] = [j['Timestamp'] for j in json_str['Items']]
      if isinstance(json_str['Items'][0]['Value'], dict):
          tagData[tag] =  [j['Value']['Name'] if isinstance(j['Value'], dict) else j['Value'] for j in json_str['Items']]
      else:
          tagData[tag] = [j['Value'] for j in json_str['Items']]
  df = pd.DataFrame(tagData, columns = tagData.keys())
  df['Date'] = pd.to_datetime(df['Date']).dt.tz_convert('US/Mountain')
  return df  #, web_ids

def pi2pd_rawData(tags,
                start_date=par.start_date, end_date=par.end_date,
                pi_vault_dict='webapi',
                custom_config=None,
                platform='databricks'):
  """Call PI Cloud API's stream controller to get tags in with their original frequency

  Params:
    tags: (list)                : maximum 11 tags at a time: 
    
    start_date (string)          : start date, format: "%Y-%m-%d"
    
    end_date   (string)          : end date, format: "%Y-%m-%d"
        
    pi_vault_dict(Dictionary)   : key_vault_dict(string) key name for webapi in azure_valuet_cred dictionary     
    
    custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead
  
  """ 
  #   start_date='2023-07-13'
  #   end_date='2023-08-10'
  #   tags=PI_WEB_API_TAGS.keys()
  #   pi_vault_dict='webapi'
  #   platform='local'

  import requests, urllib, json
  if isinstance(start_date,str):
    start_date   = dt.datetime.strptime(start_date, "%Y-%m-%d") ###'2021-12-31', "%Y-%m-%d"
  
  if isinstance(end_date, str):
    end_date     = dt.datetime.strptime(end_date,   "%Y-%m-%d") ###'2021-12-31', "%Y-%m-%d"

  if not isinstance(tags, list):tags=[tags]

  c = cred_strings(key_vault_dict=pi_vault_dict,
                   custom_config=custom_config,
                   platform=platform)
  accessToken = c.pi_server_connector()
  web_ids     = get_web_ids(accessToken, tags)
  
  data_entries = []
  for tag_name in tags:
    webID = web_ids[tag_name]
    print('tag=',tag_name,",webID=",f"{webID[:5]}...{webID[20:25]}...{webID[-5:]}")
    if webID is not None:
      query_dict = {'startTime':start_date, 'endTime':end_date}
      url = 'https://svc.apiproxy.exxonmobil.com/KRLPIV01/v1/piwebapi/streams/'+webID+'/recorded?'+urllib.parse.urlencode(query_dict)
      headers = {
        'Authorization':'Bearer ' + accessToken,
        'Content-Type':'application/json'}
      response = requests.request("GET", url, headers=headers)
      if response.status_code == 200:
        json_str = json.loads(response.text)
        # print(json_str)
        if 'Items' in json_str:
          for entry in json_str['Items']:
            timestamp = entry['Timestamp']
            value = entry['Value']
            if isinstance(value, dict):
              if value.get('Name') == 'Bad':
                value = None
              else:
                value = value.get('Value')
            data_entries.append({'Timestamp': timestamp, tag_name: value})
  df = pd.DataFrame(data_entries)
  df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_convert('US/Mountain')
  return df

def pi2pd_seconds(tags,
                start_date=par.start_date, end_date=par.end_date,
                pi_vault_dict='webapi',
                custom_config=None,
                platform='databricks'):
  """Call PI Cloud API's stream controller to get tags in every second
  maximum 11 tags at a time get pi data by second

  Params:
    tags: (list)                : maximum 11 tags at a time: 
    
    start_date (string)          : start date, format: "%Y-%m-%d"
    
    end_date   (string)          : end date, format: "%Y-%m-%d"
        
    pi_vault_dict(Dictionary)   : key_vault_dict(string) key name for webapi in azure_valuet_cred dictionary   
    
    custom_config(dict/filePath): a dictionary of configuration credentials or path of a yaml file, if it is not provided, dsToolbox.config_dict will be used instead
    
  """ 
  #   start_date='2023-07-13'
  #   end_date='2023-08-10'
  #   tags=PI_WEB_API_TAGS.keys()
  #   pi_vault_dict='webapi'
  #   platform='local'

  start_date   = dt.datetime.strptime(start_date, "%Y-%m-%d") ###'2021-12-31', "%Y-%m-%d"
  end_date     = dt.datetime.strptime(end_date,   "%Y-%m-%d") ###'2021-12-31', "%Y-%m-%d"

  start = start_date
  end = start + dt.timedelta(days=1)
  ret = pd.DataFrame()
  while(end <= end_date):
    print(f"getting data between {start} and {end}")
    try:
        
      pi_web_df = pi2pd_interpolate(tags,
                                    start_date=start, end_date=end,
                                    interval = '1s',
                                    pi_vault_dict=pi_vault_dict, 
                                    custom_config=custom_config,
                                    platform=platform)
      print("Done")
    except:
      print("Skipped")
      start = end
      end += dt.timedelta(days=1)
      continue
    ret = pd.concat([ret, pi_web_df], ignore_index=True)
    start = end
    end += dt.timedelta(days=1)
  # ret.drop_duplicates(inplace=True) 
  return ret

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ---------------------------- ------------------------------------    