import yaml
import pyodbc
import urllib.parse
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
from typing import Dict, Tuple, Any, Optional
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
        Create SQL Server connection using pyodbc with Windows authentication.
        
        Args:
            server_id (str): Server identifier from config
            
        Returns:
            Tuple[Engine, Dict]: SQLAlchemy engine and connection parameters
        """
        try:
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
            
            # Create SQLAlchemy engine from connection string
            db_params = urllib.parse.quote_plus(connection_string)
            engine = create_engine(f'mssql+pyodbc:///?odbc_connect={db_params}')
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT @@VERSION"))
                version = result.fetchone()[0]
                print(f"Connected to SQL Server via pyodbc")
            
            # Return connection parameters as dict
            connection_params = {
                'db_server': db_server,
                'driver': 'ODBC Driver 17 for SQL Server',
                'trusted_connection': mssql_config.get('trusted_connection', True),
                'trust_server_certificate': mssql_config.get('trust_server_certificate', True),
                'database': mssql_config.get('database', None),
                'connection_string': connection_string
            }
            
            return engine, connection_params
            
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
            
            connection_string = (
                f"postgresql://{pg_config['user']}:{pg_config['password']}"
                f"@{pg_config['host']}:{pg_config.get('port', 5432)}"
                f"/{pg_config['database']}"
            )
            
            # Add SSL parameters if specified in config
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
            
            # Add SSL parameters if specified in config
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
            
            engine = create_engine(connection_string)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                db_version = result.fetchone()[0]
                print(f"Connected to PostgreSQL via psycopg2: {db_version[:50]}...")
            
            # Return connection parameters as dict
            connection_params = {
                'host': pg_config['host'],
                'port': pg_config.get('port', 5432),
                'database': pg_config['database'],
                'user': pg_config['user'],
                'password': pg_config['password'],
                'sslmode': pg_config.get('sslmode', None),
                'sslcert': pg_config.get('sslcert', None),
                'sslkey': pg_config.get('sslkey', None),
                'sslrootcert': pg_config.get('sslrootcert', None),
                'connection_string': connection_string
            }
            
            return engine, connection_params
            
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
                f"/{pg_config['database']}"
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
                'database': pg_config['database'],
                'user': pg_config['user'],
                'password': pg_config['password'],
                'sslmode': pg_config.get('sslmode', None),
                'sslcert': pg_config.get('sslcert', None),
                'sslkey': pg_config.get('sslkey', None),
                'sslrootcert': pg_config.get('sslrootcert', None),
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
    
    @contextmanager
    def get_connection_context(self, db_type: str, server_id: str, use_sqlalchemy: bool = False):
        """
        Context manager for database connections with automatic cleanup.
        
        Args:
            db_type (str): 'mssql', 'postgresql', or 'incorta'
            server_id (str): Server identifier from config
            use_sqlalchemy (bool): Whether to use SQLAlchemy method (for mssql/postgresql)
        
        Yields:
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
            
            yield engine, db_params
            
        finally:
            if engine:
                try:
                    if hasattr(engine, 'dispose'):
                        engine.dispose()
                        print(f"Engine for {db_type} disposed")
                    elif hasattr(engine, 'close'):
                        engine.close()
                        print(f"Connection to {db_type} closed")
                except Exception as e:
                    print(f"Error closing connection: {e}")


# Example usage
if __name__ == "__main__":
    # Example of how to use the standardized interface
    db_manager = DatabaseConnectionManager('config.yaml')
    
    # All methods now return (engine, db_params) consistently
    try:
        # SQL Server with Windows authentication
        engine, params = db_manager.get_mssql_pyodbc_winauth('server1')
        print(f"SQL Server connected: {params['db_server']}")
        
        # PostgreSQL with credentials (SQLAlchemy)
        engine, params = db_manager.get_pg_alchemy_creds('pg_server1')
        print(f"PostgreSQL connected: {params['host']}:{params['port']}")
        
        # Incorta
        engine, params = db_manager.get_incorta()
        print(f"Incorta connected: {params['host']}")
        
        # Using context manager
        with db_manager.get_connection_context('mssql', 'server1') as (engine, params):
            with engine.connect() as conn:
                result = conn.execute(text("SELECT GETDATE()"))
                print(f"Current time: {result.fetchone()[0]}")
                
    except Exception as e:
        print(f"Connection error: {e}")