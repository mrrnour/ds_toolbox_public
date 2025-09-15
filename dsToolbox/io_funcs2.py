import sys
import yaml
import pyodbc
import urllib.parse
from sqlalchemy import create_engine, text
import psycopg2
from contextlib import contextmanager
from typing import List, Dict, Tuple, Any, Optional
import logging

class DatabaseConnectionManager:
    """
    Unified database connection manager for SQL Server, PostgreSQL, and Incorta.
    All connection methods return (engine, db_params) for consistency.
    """
    
    def __init__(self, config_file: str):
        """
        Initialize with configuration file.
        
        Args:
            config_file (str): Path to YAML configuration file
        """
        self.config = self._load_config(config_file)
        
    def _load_config(self, config_file: str) -> Dict:
        """Load and validate configuration file."""
        try:
            with open(config_file, 'r') as stream:
                config = yaml.safe_load(stream)
            print(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            print(f"Configuration file not found: {config_file}")
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            raise
    
    def get_mssql_pyodbc_winauth(self, server_id: str) -> Tuple[Any, Dict]:
        """
        Create SQL Server connection using direct pyodbc with Windows authentication.
        
        Args:
            server_id (str): Server identifier from config
            
        Returns:
            Tuple[Connection, Dict]: Direct pyodbc connection and connection parameters
        """
        try:
            import pyodbc
            
            mssql_config = self.config['mssql_servers'][server_id]
            db_server = mssql_config['db_server']
            
            # Build connection string
            connection_string = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER={db_server};"
            )
            
            # Add trusted connection if specified in config
            if mssql_config.get('trusted_connection', True):
                connection_string += "Trusted_Connection=yes;"
            
            # Add trust server certificate if specified in config
            if mssql_config.get('trust_server_certificate', True):
                connection_string += "TrustServerCertificate=yes;"
            
            # Add database if specified
            if 'database' in mssql_config:
                connection_string += f"DATABASE={mssql_config['database']};"
            
            # Create direct pyodbc connection
            conn = pyodbc.connect(connection_string)
            
            # Test connection
            cursor = conn.cursor()
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()[0]
            print(f"Connected to SQL Server via direct pyodbc")
            cursor.close()
            
            # Return connection parameters as dict
            connection_params = {
                'db_server': db_server,
                'driver': 'ODBC Driver 17 for SQL Server',
                'trusted_connection': mssql_config.get('trusted_connection', True),
                'trust_server_certificate': mssql_config.get('trust_server_certificate', True),
                'database': mssql_config.get('database', None),
                'connection_string': connection_string
            }
            
            return conn, connection_params
            
        except KeyError as e:
            print(f"Missing configuration key: {e}")
            raise
        except Exception as e:
            print(f"SQL Server pyodbc connection error: {e}")
            raise

    def get_mssql_alchemy_winauth(self, server_id: str) -> Tuple[Any, Dict]:
        """
        Create SQL Server connection using SQLAlchemy with Windows authentication.
        
        Args:
            server_id (str): Server identifier from config
            
        Returns:
            Tuple[Engine, Dict]: SQLAlchemy engine and connection parameters
        """
        try:
            mssql_config = self.config['mssql_servers'][server_id]
            db_server = mssql_config['db_server']
            
            # Build connection string
            driver_params = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={db_server};"
            )
            
            # Add trusted connection if specified in config
            if mssql_config.get('trusted_connection', True):
                driver_params += "Trusted_Connection=yes;"
            
            # Add trust server certificate if specified in config
            if mssql_config.get('trust_server_certificate', True):
                driver_params += "TrustServerCertificate=yes;"
            
            # Add database if specified
            if 'database' in mssql_config:
                driver_params += f"DATABASE={mssql_config['database']};"
            
            db_params_encoded = urllib.parse.quote_plus(driver_params)
            engine = create_engine(f'mssql+pyodbc:///?odbc_connect={db_params_encoded}')
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT @@VERSION"))
                version = result.fetchone()[0]
                print(f"Connected to SQL Server via SQLAlchemy")
            
            # Return connection parameters as dict
            connection_params = {
                'db_server': db_server,
                'driver': 'ODBC Driver 17 for SQL Server',
                'trusted_connection': mssql_config.get('trusted_connection', True),
                'trust_server_certificate': mssql_config.get('trust_server_certificate', True),
                'database': mssql_config.get('database', None),
                'connection_string': driver_params
            }
            
            return engine, connection_params
            
        except KeyError as e:
            print(f"Missing configuration key: {e}")
            raise
        except Exception as e:
            print(f"SQLAlchemy SQL Server connection error: {e}")
            raise
    
    def get_pg_psycopg2_creds(self, server_id: str) -> Tuple[Any, Dict]:
        """
        Create PostgreSQL connection using username/password credentials (psycopg2 method).
        
        Args:
            server_id (str): Server identifier from config
            
        Returns:
            Tuple[Engine, Dict]: SQLAlchemy engine and connection parameters
        """
        try:
            
            pg_config = self.config['postgresql_servers'][server_id]
            
            # Build connection parameters for psycopg2
            conn_params = {
                'host': pg_config['host'],
                'port': pg_config.get('port', 5432),
                'database': pg_config.get('database', 'postgres'),
                'user': pg_config['user'],
                'password': pg_config['password']
            }
            
            # Add SSL parameters if specified in config
            if 'sslmode' in pg_config:
                conn_params['sslmode'] = pg_config['sslmode']
            if 'sslcert' in pg_config:
                conn_params['sslcert'] = pg_config['sslcert']
            if 'sslkey' in pg_config:
                conn_params['sslkey'] = pg_config['sslkey']
            if 'sslrootcert' in pg_config:
                conn_params['sslrootcert'] = pg_config['sslrootcert']
            
            # Create psycopg2 connection
            psycopg2_conn = psycopg2.connect(**conn_params)
            
            # Test connection
            with psycopg2_conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                db_version = cursor.fetchone()[0]
                print(f"Connected to PostgreSQL via psycopg2: {db_version[:50]}...")
            
            # Build connection string for reference
            connection_string = (
                f"postgresql://{pg_config['user']}:{pg_config['password']}"
                f"@{pg_config['host']}:{pg_config.get('port', 5432)}"
                f"/{pg_config.get('database', 'postgres')}"
            )
            
            # Add SSL parameters to connection string if specified
            ssl_params = []
            if 'sslmode' in pg_config:
                ssl_params.append(f"sslmode={pg_config['sslmode']}")
            if 'sslcert' in pg_config:
                ssl_params.append(f"sslcert={pg_config['sslcert']}")
            if 'sslkey' in pg_config:
                ssl_params.append(f"sslkey={pg_config['sslkey']}")
            if 'sslrootcert' in pg_config:
                ssl_params.append(f"sslrootcert={pg_config['sslrootcert']}")
            
            if ssl_params:
                connection_string += "?" + "&".join(ssl_params)
            
            # Return connection parameters as dict
            connection_params = {
                'host': pg_config['host'],
                'port': pg_config.get('port', 5432),
                'database': pg_config.get('database', 'postgres'),
                'user': pg_config['user'],
                'password': pg_config['password'],
                'sslmode': pg_config.get('sslmode', None),
                'sslcert': pg_config.get('sslcert', None),
                'sslkey': pg_config.get('sslkey', None),
                'sslrootcert': pg_config.get('sslrootcert', None),
                'connection_string': connection_string
            }
            
            return psycopg2_conn, connection_params
            
        except KeyError as e:
            print(f"Missing PostgreSQL configuration key: {e}")
            raise
        except Exception as e:
            print(f"PostgreSQL psycopg2 connection error: {e}")
            raise

    def get_pg_alchemy_creds(self, server_id: str) -> Tuple[Any, Dict]:
        """
        Create PostgreSQL connection using username/password credentials (SQLAlchemy method).
        
        Args:
            server_id (str): Server identifier from config
            
        Returns:
            Tuple[Engine, Dict]: SQLAlchemy engine and connection parameters
        """
        try:
            pg_config = self.config['postgresql_servers'][server_id]
            
            connection_string = (
                f"postgresql://{pg_config['user']}:{pg_config['password']}"
                f"@{pg_config['host']}:{pg_config.get('port', 5432)}"
                f"/{pg_config.get('database', 'postgres')}"
            )
            
            engine = create_engine(connection_string)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                db_version = result.fetchone()[0]
                print(f"Connected to PostgreSQL via SQLAlchemy: {db_version[:50]}...")
            
            # Return connection parameters as dict
            connection_params = {
                'host': pg_config['host'],
                'port': pg_config.get('port', 5432),
                'database': pg_config.get('database', 'postgres'),
                'user': pg_config['user'],
                'password': pg_config['password'],
                # 'sslmode': pg_config.get('sslmode', None),
                # 'sslcert': pg_config.get('sslcert', None),
                # 'sslkey': pg_config.get('sslkey', None),
                # 'sslrootcert': pg_config.get('sslrootcert', None),
                'connection_string': connection_string
            }
            
            return engine, connection_params
            
        except KeyError as e:
            print(f"Missing PostgreSQL configuration key: {e}")
            raise
        except Exception as e:
            print(f"SQLAlchemy PostgreSQL connection error: {e}")
            raise
    
    def get_incorta(self) -> Tuple[Any, Dict]:
        """
        Create Incorta connection wrapped in SQLAlchemy.
        
        Returns:
            Tuple[Engine, Dict]: SQLAlchemy engine and connection parameters
        """
        try:
            incorta_config = self.config['incorta_server']
            
            connection_string = (
                f"DRIVER={{Incorta ODBC Driver}};"
                f"HOST={incorta_config['host']};"
                f"PORT={incorta_config['port']};"
                f"DATABASE={incorta_config['database']};"
                f"UID={incorta_config['user']};"
                f"PWD={incorta_config['password']};"
            )
            
            # Create SQLAlchemy engine from ODBC connection string
            db_params_encoded = urllib.parse.quote_plus(connection_string)
            engine = create_engine(f'mssql+pyodbc:///?odbc_connect={db_params_encoded}')
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                print(f"Connected to Incorta: {incorta_config['host']}")
            
            # Return connection parameters as dict
            connection_params = {
                'host': incorta_config['host'],
                'port': incorta_config['port'],
                'database': incorta_config['database'],
                'user': incorta_config['user'],
                'password': incorta_config['password'],
                'driver': 'Incorta ODBC Driver',
                'connection_string': connection_string
            }
            
            return engine, connection_params
            
        except KeyError as e:
            print(f"Missing Incorta configuration key: {e}")
            raise
        except Exception as e:
            print(f"Incorta connection error: {e}")
            raise
    
    def get_connection_context(self, db_type: str, server_id: str, use_sqlalchemy: bool = False):
        """
        Get database connection with automatic cleanup capability.
        
        Args:
            db_type (str): 'mssql', 'postgresql', or 'incorta'
            server_id (str): Server identifier from config
            use_sqlalchemy (bool): Whether to use SQLAlchemy method (for mssql/postgresql)
        
        Returns:
            Tuple[Engine, Dict]: Engine and connection parameters
        """
        engine = None
        try:
            if db_type == 'mssql':
                if use_sqlalchemy:
                    engine, db_params = self.get_mssql_alchemy_winauth(server_id)
                else:
                    engine, db_params = self.get_mssql_pyodbc_winauth(server_id)
            elif db_type == 'postgresql':
                if use_sqlalchemy:
                    engine, db_params = self.get_pg_alchemy_creds(server_id)
                else:
                    engine, db_params = self.get_pg_psycopg2_creds(server_id)
            elif db_type == 'incorta':
                engine, db_params = self.get_incorta()
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            return engine, db_params  # Changed from 'yield' to 'return'
            
        except Exception as e:
            # Clean up engine if there was an error during creation
            if engine:
                try:
                    if hasattr(engine, 'dispose'):
                        engine.dispose()
                    elif hasattr(engine, 'close'):
                        engine.close()
                except Exception as cleanup_error:
                    print(f"Error during cleanup: {cleanup_error}")
            raise

def query_psycopg2(query: str, engine_psycopg2) -> List[Dict]:
    """
    Execute a query using a psycopg2 connection.
    
    Args:
        query (str): SQL query to execute
        engine_psycopg2: psycopg2 connection object
        
    Returns:
        List[Dict]: Query results as list of dictionaries
    """
    try:
        from psycopg2.extras import RealDictCursor
        
        with engine_psycopg2.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            
            # Handle different query types
            if query.strip().upper().startswith(('SELECT', 'WITH', 'SHOW', 'EXPLAIN')):
                results = cursor.fetchall()
                # Convert RealDictRow to regular dict
                return [dict(row) for row in results]
            else:
                # For INSERT, UPDATE, DELETE, etc.
                engine_psycopg2.commit()
                return [{"rows_affected": cursor.rowcount}]
                
    except Exception as e:
        # Rollback on error
        engine_psycopg2.rollback()
        print(f"Query execution error: {e}")
        raise

# ##TODO: check AzureConnectionManager and retire db connectors from io_funcs.py
# class AzureConnectionManager:
#     """
#     Unified Azure connection manager for Blob Storage, Synapse, and PI Server.
#     All connection methods return consistent outputs for various Azure services.
#     """
    
#     def __init__(self, config_file: str):
#         """
#         Initialize with configuration file.
        
#         Args:
#             config_file (str): Path to YAML configuration file
#         """
#         self.config = self._load_config(config_file)
        
#     def _load_config(self, config_file: str) -> Dict:
#         """Load and validate configuration file."""
#         try:
#             with open(config_file, 'r') as stream:
#                 config = yaml.safe_load(stream)
#             print(f"Configuration loaded from {config_file}")
#             return config
#         except FileNotFoundError:
#             print(f"Configuration file not found: {config_file}")
#             raise
#         except yaml.YAMLError as e:
#             print(f"Error parsing YAML file: {e}")
#             raise
    
#     def _get_key_vault_credentials(self, storage_account: str, platform: str = 'databricks') -> str:
#         """
#         Retrieve credentials from Azure Key Vault.
        
#         Args:
#             storage_account (str): Storage account identifier
#             platform (str): Platform type (default: 'databricks')
            
#         Returns:
#             str: Retrieved password/secret
#         """
#         try:
#             key_vault_dictS = self.config.get('key_vault_dictS', {})
#             KV_access_local = self.config.get('KV_access_local')
#             azure_ml_appID = self.config.get('azure_ml_appID')
            
#             if storage_account not in key_vault_dictS:
#                 raise KeyError(f"Storage account '{storage_account}' not found in key vault configuration")
                
#             cred_dict = key_vault_dictS[storage_account]
#             key_vault_name = cred_dict.get('key_vault_name')
#             secret_name = cred_dict.get('secret_name')
            
#             if not key_vault_name or not secret_name:
#                 raise KeyError("Missing key_vault_name or secret_name in configuration")
            
#             # Note: This assumes fetch_key_value function exists in the environment
#             password = fetch_key_value(
#                 key_vault_name,
#                 secret_name,
#                 azure_ml_appID,
#                 KV_access_local,
#                 platform
#             )
            
#             return password
            
#         except Exception as e:
#             print(f"Error retrieving key vault credentials: {e}")
#             raise
    
#     def get_blob_storage_connection(self, storage_account: str, 
#                                   container: str, 
#                                   filename: str = "", 
#                                   platform: str = 'databricks') -> Tuple[str, Dict]:
#         """
#         Create Azure Blob Storage connection parameters.
        
#         Args:
#             storage_account (str): Storage account identifier from config
#             container (str): Blob container name
#             filename (str): Optional filename for blob path
#             platform (str): Platform type (default: 'databricks')
            
#         Returns:
#             Tuple[str, Dict]: Connection string and connection parameters
#         """
#         try:
#             password = self._get_key_vault_credentials(storage_account, platform)
            
#             # Build connection components
#             blob_host = f"fs.azure.account.key.{storage_account}.blob.core.windows.net"
#             path = f'{container}@{storage_account}'
#             blob_path = f"wasbs://{path}.blob.core.windows.net/{filename}"
#             blob_connection_str = (
#                 f'DefaultEndpointsProtocol=https;'
#                 f'AccountName={storage_account};'
#                 f'AccountKey={password};'
#                 f'EndpointSuffix=core.windows.net'
#             )
            
#             # Test connection would go here in a real implementation
#             print(f"Blob storage connection configured for account: {storage_account}")
            
#             connection_params = {
#                 'storage_account': storage_account,
#                 'container': container,
#                 'filename': filename,
#                 'blob_host': blob_host,
#                 'blob_path': blob_path,
#                 'connection_string': blob_connection_str,
#                 'platform': platform
#             }
            
#             return blob_connection_str, connection_params
            
#         except KeyError as e:
#             print(f"Missing Blob Storage configuration key: {e}")
#             raise
#         except Exception as e:
#             print(f"Blob Storage connection error: {e}")
#             raise
    
#     def get_spark_connection(self, storage_account: str, 
#                            platform: str = 'databricks') -> Tuple[str, Dict]:
#         """
#         Create Spark/Data Lake connection parameters.
        
#         Args:
#             storage_account (str): Storage account identifier from config
#             platform (str): Platform type (default: 'databricks')
            
#         Returns:
#             Tuple[str, Dict]: Spark host and connection parameters
#         """
#         try:
#             key_vault_dictS = self.config.get('key_vault_dictS', {})
#             password = self._get_key_vault_credentials(storage_account, platform)
            
#             # Note: Original code used key_vault_name instead of storage_account
#             # This might be a bug in the original - keeping for compatibility
#             if storage_account in key_vault_dictS:
#                 key_vault_name = key_vault_dictS[storage_account].get('key_vault_name')
#             else:
#                 key_vault_name = storage_account
                
#             spark_host = f"fs.azure.account.key.{key_vault_name}.dfs.core.windows.net"
            
#             print(f"Spark connection configured for: {key_vault_name}")
            
#             connection_params = {
#                 'storage_account': storage_account,
#                 'key_vault_name': key_vault_name,
#                 'spark_host': spark_host,
#                 'platform': platform
#             }
            
#             return spark_host, connection_params
            
#         except Exception as e:
#             print(f"Spark connection error: {e}")
#             raise
    
#     def get_synapse_connection(self, storage_account: str, 
#                              platform: str = 'databricks') -> Tuple[Any, Dict]:
#         """
#         Create Synapse SQL connection using SQLAlchemy.
        
#         Args:
#             storage_account (str): Storage account identifier from config
#             platform (str): Platform type (default: 'databricks')
            
#         Returns:
#             Tuple[Engine, Dict]: SQLAlchemy engine and connection parameters
#         """
#         try:
#             synapse_cred_dict = self.config.get('synapse_cred_dict', {})
#             password = self._get_key_vault_credentials(storage_account, platform)
            
#             # Get Synapse configuration
#             hostname = synapse_cred_dict.get('hostname')
#             database = synapse_cred_dict.get('database')
#             port = synapse_cred_dict.get('port')
#             username = synapse_cred_dict.get('username')
#             driver = synapse_cred_dict.get('driver')
#             driver_odbc = synapse_cred_dict.get('driver_odbc')
            
#             if not all([hostname, database, port, username]):
#                 raise KeyError("Missing required Synapse configuration parameters")
            
#             # Build JDBC URL and properties
#             jdbc_url = f"jdbc:sqlserver://{hostname}:{port};database={database}"
            
#             properties = {
#                 "user": username,
#                 "password": password,
#                 "driver": driver
#             }
            
#             # Build ODBC connection string
#             odbc_connector = (
#                 f"DRIVER={driver_odbc};"
#                 f"SERVER={hostname};"
#                 f"PORT={port};"
#                 f"DATABASE={database};"
#                 f"UID={username};"
#                 f"PWD={password};"
#                 f"MARS_Connection=yes"
#             )
            
#             # Create SQLAlchemy engine
#             db_params_encoded = urllib.parse.quote_plus(odbc_connector)
#             engine = create_engine(f'mssql+pyodbc:///?odbc_connect={db_params_encoded}')
            
#             # Test connection
#             with engine.connect() as conn:
#                 result = conn.execute(text("SELECT @@VERSION"))
#                 version = result.fetchone()[0]
#                 print(f"Connected to Synapse: {hostname}")
            
#             connection_params = {
#                 'storage_account': storage_account,
#                 'hostname': hostname,
#                 'database': database,
#                 'port': port,
#                 'username': username,
#                 'driver': driver,
#                 'driver_odbc': driver_odbc,
#                 'jdbc_url': jdbc_url,
#                 'properties': properties,
#                 'odbc_connector': odbc_connector,
#                 'platform': platform
#             }
            
#             return engine, connection_params
            
#         except KeyError as e:
#             print(f"Missing Synapse configuration key: {e}")
#             raise
#         except Exception as e:
#             print(f"Synapse connection error: {e}")
#             raise
    
#     def get_pi_server_connection(self, storage_account: str, 
#                                platform: str = 'databricks') -> Tuple[str, Dict]:
#         """
#         Create PI Server OAuth connection and retrieve access token.
        
#         Args:
#             storage_account (str): Storage account identifier from config
#             platform (str): Platform type (default: 'databricks')
            
#         Returns:
#             Tuple[str, Dict]: Access token and connection parameters
#         """
#         try:
#             pi_server_dict = self.config.get('pi_server', {})
#             password = self._get_key_vault_credentials(storage_account, platform)
            
#             # Get PI Server configuration
#             url = pi_server_dict.get('url')
#             grant_type = pi_server_dict.get('grant_type')
#             client_id = pi_server_dict.get('client_id')
#             client_secret_scope = pi_server_dict.get('client_secret')
            
#             if not all([url, grant_type, client_id, client_secret_scope]):
#                 raise KeyError("Missing required PI Server configuration parameters")
            
#             # Prepare OAuth request
#             oauth_payload = {
#                 'grant_type': grant_type,
#                 'client_id': client_id,
#                 'scope': client_secret_scope,
#                 'client_secret': password
#             }
            
#             # Make OAuth request
#             import requests
#             oauth_response = requests.post(url, data=oauth_payload)
#             oauth_response.raise_for_status()
            
#             access_token = oauth_response.json().get('access_token')
            
#             if not access_token:
#                 raise ValueError("Failed to retrieve access token from PI Server")
            
#             print(f"PI Server OAuth token retrieved successfully")
            
#             connection_params = {
#                 'storage_account': storage_account,
#                 'url': url,
#                 'grant_type': grant_type,
#                 'client_id': client_id,
#                 'client_secret_scope': client_secret_scope,
#                 'access_token': access_token,
#                 'platform': platform
#             }
            
#             return access_token, connection_params
            
#         except KeyError as e:
#             print(f"Missing PI Server configuration key: {e}")
#             raise
#         except Exception as e:
#             print(f"PI Server connection error: {e}")
#             raise
    
#     def get_connection_context(self, connection_type: str, 
#                              storage_account: str, 
#                              platform: str = 'databricks', 
#                              **kwargs) -> Tuple[Any, Dict]:
#         """
#         Get Azure service connection with automatic cleanup capability.
        
#         Args:
#             connection_type (str): 'blob', 'spark', 'synapse', or 'pi_server'
#             storage_account (str): Storage account identifier from config
#             platform (str): Platform type (default: 'databricks')
#             **kwargs: Additional parameters specific to connection type
        
#         Returns:
#             Tuple[Any, Dict]: Connection object/string and connection parameters
#         """
#         try:
#             if connection_type == 'blob':
#                 container = kwargs.get('container')
#                 filename = kwargs.get('filename', '')
#                 if not container:
#                     raise ValueError("Container name is required for blob connections")
#                 return self.get_blob_storage_connection(storage_account, container, filename, platform)
                
#             elif connection_type == 'spark':
#                 return self.get_spark_connection(storage_account, platform)
                
#             elif connection_type == 'synapse':
#                 return self.get_synapse_connection(storage_account, platform)
                
#             elif connection_type == 'pi_server':
#                 return self.get_pi_server_connection(storage_account, platform)
                
#             else:
#                 raise ValueError(f"Unsupported connection type: {connection_type}")
                
#         except Exception as e:
#             print(f"Error creating {connection_type} connection: {e}")
#             raise

# def get_dbutils():
#   import IPython
#   dbutils = IPython.get_ipython().user_ns["dbutils"]
#   return dbutils

# def get_secret_KVUri(key_vault_name, secret_name, credential):
#     from azure.keyvault.secrets import SecretClient    
#     KVUri = f"https://{key_vault_name}.vault.azure.net"
#     client = SecretClient(vault_url=KVUri, credential=credential)
#     secret = client.get_secret(secret_name).value
#     return secret

# def fetch_key_value(key_vault_name, secret_name, azure_ml_appID, KV_access_local, platform='databricks'):
#     if platform == 'databricks':
#         # print('i am databricks run')
#         dbutils = get_dbutils()
#         return dbutils.secrets.get(scope=key_vault_name, key=secret_name)

#     if platform == 'aml':
#         print("using azure ML and managed identity authentication")
#         if azure_ml_appID is None:
#             sys.exit("Identity Application ID is not provided")
        
#         from azure.identity import ManagedIdentityCredential
#         client_id = f"{azure_ml_appID}"
#         credential = ManagedIdentityCredential(client_id=client_id)
#         credential.get_token("https://vault.azure.net/.default")
#         return get_secret_KVUri(key_vault_name, secret_name, credential=credential)

#     if platform in ['local', 'vm_docker']:
#         print('i am locally run')
        
#         try:
#             import os
#             if os.environ.get('AZURE_TENANT_ID') is None:
#                 os.environ['AZURE_TENANT_ID'] = KV_access_local['secret_TenantID']
#             if os.environ.get('AZURE_CLIENT_ID') is None:
#                 os.environ['AZURE_CLIENT_ID'] = KV_access_local['secret_ClientID__prd']
#             if os.environ.get('AZURE_CLIENT_SECRET') is None:
#                 os.environ['AZURE_CLIENT_SECRET'] = KV_access_local['secret_ClientSecret__Prd']
#         except Exception as e:
#             print(f'{str(e)}')
#             if KV_access_local is None:
#                 sys.exit("""set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET environment variables 
#                         or provide KV_access_local dictionary in config.yml file to extract them """)
        
#         from azure.identity import DefaultAzureCredential
#         return get_secret_KVUri(key_vault_name, secret_name, credential=DefaultAzureCredential())

#     # This should not be reached, but keeping for safety
#     return None
