"""
Azure Operations Manager for Data Science Toolbox I/O Operations
================================================================

Enhanced Azure operations manager for Synapse, Blob Storage, and Delta Tables.
Provides comprehensive Azure cloud operations with automatic environment
detection and platform-adaptive functionality (Databricks vs Local execution).

Classes:
--------
- AzureManager: Azure cloud services operations manager

Dependencies:
------------
- azure-storage-blob: For blob storage operations
- azure-synapse: For Synapse analytics operations
- sqlalchemy: For database connections

Author: Data Science Toolbox Contributors  
License: MIT License
"""

import logging
from typing import Dict, Tuple, Optional

# Internal imports
try:
    from .config import ConfigurationManager
except ImportError:
    try:
        from dsToolbox.io.config import ConfigurationManager
    except ImportError:
        # Fallback for backward compatibility
        ConfigurationManager = None
        logging.warning("ConfigurationManager not available - some functionality may be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureManager:
    """
    Enhanced Azure operations manager for Synapse, Blob Storage, and Delta Tables.
    
    This class provides comprehensive Azure cloud operations with automatic environment
    detection and platform-adaptive functionality (Databricks vs Local execution).
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize the Azure manager."""
        self.config_manager = config_manager or (ConfigurationManager() if ConfigurationManager else None)
    
    def get_synapse_connection_details(self, key_vault_identifier: str = 'azure_synapse') -> Tuple[str, Dict[str, str]]:
        """
        Get Synapse connection details for reuse - preferred approach for multiple queries.
        
        Parameters
        ----------
        key_vault_identifier : str, default='azure_synapse'
            Key vault configuration identifier
            
        Returns
        -------
        tuple
            (connection_url, connection_properties) for Spark JDBC operations
            
        Examples
        --------
        >>> azure_mgr = AzureManager()
        >>> url, props = azure_mgr.get_synapse_connection_details('azure_synapse')
        >>> df = spark.read.jdbc(table="(SELECT * FROM table) query", url=url, properties=props)
        """
        if not self.config_manager:
            raise RuntimeError("ConfigurationManager required for Synapse connections")
            
        try:
            synapse_config = self.config_manager.get_database_configuration('synapse_credentials')
            
            server = synapse_config.get('server')
            database = synapse_config.get('database')
            username = synapse_config.get('username', '')
            
            if not server or not database:
                raise ValueError("Synapse server and database must be configured")
            
            # Get password from configuration or key vault
            password = synapse_config.get('password', '')
            if not password:
                # Try to get from key vault if configured
                password = self._get_azure_key_vault_secret(key_vault_identifier)
            
            jdbc_url = f"jdbc:sqlserver://{server}:1433;database={database}"
            connection_properties = {
                "user": username,
                "password": password,
                "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
            }
            
            return jdbc_url, connection_properties
            
        except Exception as e:
            raise Exception(f"Failed to get Synapse connection details: {e}")
    
    def _get_azure_key_vault_secret(self, key_vault_identifier: str) -> str:
        """Get secret from Azure Key Vault."""
        # This would integrate with Azure Key Vault SDK
        # Implementation depends on specific Azure setup
        raise NotImplementedError("Azure Key Vault integration not implemented in this version")


# Export all classes for external use
__all__ = [
    'AzureManager'
]