"""
AWS Services Manager for Data Science Toolbox I/O Operations
============================================================

Comprehensive AWS services manager for S3, Athena, and other AWS operations.
Provides unified AWS operations including S3 storage management, Athena query 
execution, and data transfer operations commonly used in data science workflows.

Classes:
--------
- AWSManager: Comprehensive AWS services manager

Dependencies:
------------
- boto3: Required for AWS service interactions
- pandas: For DataFrame operations

Author: Data Science Toolbox Contributors
License: MIT License
"""

import os
import logging
from typing import Dict, Any

# Third-party imports (with graceful handling)
try:
    import boto3
except ImportError as e:
    logging.warning(f"Boto3 dependency not found: {e}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSManager:
    """
    Comprehensive AWS services manager for S3, Athena, and other AWS operations.
    
    This class provides unified AWS operations including S3 storage management,
    Athena query execution, and data transfer operations commonly used in
    data science workflows.
    """
    
    def __init__(self, aws_region: str = 'us-west-2'):
        """
        Initialize AWS manager with region configuration.
        
        Parameters
        ----------
        aws_region : str, default='us-west-2'
            AWS region for service operations
        """
        self.aws_region = aws_region
        self._s3_client = None
        self._athena_client = None
        
        logger.info(f"AWSManager initialized for region: {aws_region}")
    
    @property
    def s3_client(self) -> Any:
        """Get or create S3 client with lazy initialization."""
        if self._s3_client is None:
            try:
                session = boto3.Session()
                self._s3_client = session.client('s3', region_name=self.aws_region)
                logger.info("S3 client initialized")
            except Exception as e:
                raise Exception(f"Failed to create S3 client: {e}")
        return self._s3_client
    
    @property 
    def athena_client(self) -> Any:
        """Get or create Athena client with lazy initialization."""
        if self._athena_client is None:
            try:
                self._athena_client = boto3.client('athena', region_name=self.aws_region)
                logger.info("Athena client initialized")
            except Exception as e:
                raise Exception(f"Failed to create Athena client: {e}")
        return self._athena_client
    
    def upload_file_to_s3(self, local_file_path: str, s3_bucket: str, s3_key_path: str) -> bool:
        """
        Upload local file to Amazon S3 with error handling.
        
        Parameters
        ----------
        local_file_path : str
            Path to local file to upload
        s3_bucket : str
            S3 bucket name
        s3_key_path : str
            S3 object key (path within bucket)
            
        Returns
        -------
        bool
            True if upload successful, False otherwise
            
        Examples
        --------
        >>> aws_mgr = AWSManager()
        >>> success = aws_mgr.upload_file_to_s3('/local/file.csv', 'my-bucket', 'data/file.csv')
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")
        
        try:
            self.s3_client.upload_file(local_file_path, s3_bucket, s3_key_path)
            logger.info(f"Successfully uploaded {local_file_path} to s3://{s3_bucket}/{s3_key_path}")
            return True
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    def download_file_from_s3(self, s3_bucket: str, s3_key_path: str, local_file_path: str) -> bool:
        """
        Download file from Amazon S3 to local filesystem.
        
        Parameters
        ----------
        s3_bucket : str
            S3 bucket name
        s3_key_path : str
            S3 object key to download
        local_file_path : str
            Local destination path
            
        Returns
        -------
        bool
            True if download successful, False otherwise
        """
        try:
            # Create local directory if needed
            local_dir = os.path.dirname(local_file_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)
            
            self.s3_client.download_file(s3_bucket, s3_key_path, local_file_path)
            logger.info(f"Successfully downloaded s3://{s3_bucket}/{s3_key_path} to {local_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False
    
    def clean_s3_folder(self, s3_folder_path: str) -> bool:
        """
        Clean (delete all objects in) an S3 folder.
        
        Parameters
        ----------
        s3_folder_path : str
            S3 folder path in format: s3://bucket/folder/path
            
        Returns
        -------
        bool
            True if cleanup successful, False otherwise
        """
        if not s3_folder_path.startswith('s3://'):
            raise ValueError("S3 folder path must start with 's3://'")
        
        try:
            # Parse S3 path
            path_parts = s3_folder_path.replace('s3://', '').split('/', 1)
            bucket_name = path_parts[0]
            folder_prefix = path_parts[1] if len(path_parts) > 1 else ''
            
            # Use S3 resource for object operations
            s3_resource = boto3.resource('s3', region_name=self.aws_region)
            bucket = s3_resource.Bucket(bucket_name)
            
            # Delete all objects with the prefix
            deleted_count = 0
            for obj in bucket.objects.filter(Prefix=folder_prefix):
                logger.info(f"Removing s3://{bucket_name}/{obj.key}")
                obj.delete()
                deleted_count += 1
            
            logger.info(f"Successfully cleaned S3 folder: {deleted_count} objects deleted")
            return True
            
        except Exception as e:
            logger.error(f"S3 folder cleanup failed: {e}")
            return False
    
    def execute_athena_query(self, sql_query: str, s3_output_location: str, 
                           print_query: bool = False) -> Dict[str, Any]:
        """
        Execute SQL query in Amazon Athena with result tracking.
        
        Parameters
        ----------
        sql_query : str
            SQL query to execute
        s3_output_location : str
            S3 location for query results
        print_query : bool, default=False
            Whether to print the query before execution
            
        Returns
        -------
        dict
            Query execution metadata including execution ID and status
        """
        if not sql_query or not sql_query.strip():
            raise ValueError("SQL query cannot be empty")
        
        try:
            if print_query:
                logger.info(f"Executing Athena query:\n{sql_query}")
            
            # Start query execution
            response = self.athena_client.start_query_execution(
                QueryString=sql_query.strip(),
                ResultConfiguration={'OutputLocation': s3_output_location}
            )
            
            execution_id = response['QueryExecutionId']
            
            # Wait for query completion
            query_status = 'RUNNING'
            while query_status == 'RUNNING':
                status_response = self.athena_client.get_query_execution(
                    QueryExecutionId=execution_id
                )
                query_status = status_response['QueryExecution']['Status']['State']
            
            execution_metadata = {
                'executionId': execution_id,
                'query': sql_query,
                'status': query_status,
                'date': response['ResponseMetadata']['HTTPHeaders']['date']
            }
            
            if query_status == 'SUCCEEDED':
                logger.info(f"Athena query completed successfully: {execution_id}")
            else:
                logger.error(f"Athena query failed with status: {query_status}")
            
            return execution_metadata
            
        except Exception as e:
            logger.error(f"Athena query execution failed: {e}")
            raise Exception(f"Athena query failed: {e}")


# Export all classes for external use
__all__ = [
    'AWSManager'
]