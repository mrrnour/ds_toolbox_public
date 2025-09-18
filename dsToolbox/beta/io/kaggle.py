"""
Kaggle Operations Manager for Data Science Toolbox I/O Operations
=================================================================

Enhanced Kaggle operations manager with cross-platform compatibility.
Provides Kaggle dataset download and credential management functionality 
that works across all platforms with enhanced error handling and progress tracking.

Classes:
--------
- KaggleManager: Comprehensive Kaggle dataset operations manager

Dependencies:
------------
- kaggle: Kaggle CLI for dataset downloads
- zipfile: For archive extraction (built-in)

Author: Data Science Toolbox Contributors
License: MIT License
"""

import os
import json
import shutil
import zipfile
import logging
from typing import Optional, Tuple
from pathlib import Path

# Import utility functions
try:
    from dsToolbox.utilities import FileSystemUtilities
except ImportError:
    # Graceful fallback if data utilities are not available
    FileSystemUtilities = None
    logging.warning("FileSystemUtilities not available - using fallback extraction")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KaggleManager:
    """
    Enhanced Kaggle operations manager with cross-platform compatibility.
    
    This class provides Kaggle dataset download and credential management
    functionality that works across all platforms with enhanced error handling
    and progress tracking.
    """
    
    def __init__(self):
        """Initialize the Kaggle manager."""
        self.credentials_configured = False
    
    def configure_api_credentials(self, kaggle_json_path: str) -> bool:
        """
        Configure Kaggle API credentials with comprehensive validation.
        
        Parameters
        ----------
        kaggle_json_path : str
            Path to kaggle.json credentials file
            
        Returns
        -------
        bool
            True if credentials configured successfully, False otherwise
            
        Examples
        --------
        >>> kaggle_mgr = KaggleManager()
        >>> success = kaggle_mgr.configure_api_credentials('/path/to/kaggle.json')
        """
        if not os.path.exists(kaggle_json_path):
            raise FileNotFoundError(f"Kaggle credentials file not found: {kaggle_json_path}")
        
        try:
            # Validate JSON structure
            with open(kaggle_json_path, 'r') as f:
                credentials = json.load(f)
            
            required_keys = ['username', 'key']
            if not all(key in credentials for key in required_keys):
                raise ValueError(f"Kaggle credentials must contain: {required_keys}")
            
            # Setup Kaggle directory and credentials
            kaggle_dir = os.path.expanduser('~/.kaggle')
            if not os.path.exists(kaggle_dir):
                os.makedirs(kaggle_dir, mode=0o700)
            
            credentials_dest = os.path.join(kaggle_dir, 'kaggle.json')
            shutil.copyfile(kaggle_json_path, credentials_dest)
            os.chmod(credentials_dest, 0o600)
            
            self.credentials_configured = True
            logger.info("Kaggle API credentials configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure Kaggle credentials: {e}")
            return False
    
    def download_competition_data(self, competition_name: str, destination_directory: str,
                                extract_specific_files: Optional[Tuple[str, ...]] = None,
                                exclude_files: Optional[Tuple[str, ...]] = None,
                                cleanup_zip: bool = True) -> bool:
        """
        Download and extract Kaggle competition dataset with advanced options.
        
        Parameters
        ----------
        competition_name : str
            Name of Kaggle competition
        destination_directory : str
            Local directory for extracted files
        extract_specific_files : tuple, optional
            Specific files/folders to extract
        exclude_files : tuple, optional
            Files/folders to exclude from extraction
        cleanup_zip : bool, default=True
            Whether to delete zip file after extraction
            
        Returns
        -------
        bool
            True if download and extraction successful, False otherwise
        """
        if not self.credentials_configured:
            credentials_path = os.path.expanduser('~/.kaggle/kaggle.json')
            if not os.path.exists(credentials_path):
                raise RuntimeError("Kaggle credentials not configured. Use configure_api_credentials() first.")
        
        try:
            # Create destination directory
            Path(destination_directory).mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            download_success = self._execute_kaggle_download(competition_name, destination_directory)
            if not download_success:
                return False
            
            # Extract with filtering
            zip_file = f"{competition_name}.zip"
            if FileSystemUtilities is not None:
                extraction_success = FileSystemUtilities.extract_zip_archive(
                    destination_directory, zip_file, extract_specific_files, exclude_files
                )
            else:
                # Fallback extraction
                extraction_success = self._fallback_zip_extraction(
                    destination_directory, zip_file
                )
            
            if not extraction_success:
                return False
            
            # Cleanup if requested
            if cleanup_zip:
                zip_path = os.path.join(destination_directory, zip_file)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                    logger.info(f"Cleaned up zip file: {zip_file}")
            
            logger.info(f"Kaggle competition '{competition_name}' downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")
            return False
    
    def _execute_kaggle_download(self, competition_name: str, destination_directory: str) -> bool:
        """Execute Kaggle CLI download command."""
        try:
            original_directory = os.getcwd()
            os.chdir(destination_directory)
            
            try:
                download_command = f"kaggle competitions download -c {competition_name}"
                result = os.system(download_command)
                
                if result != 0:
                    raise Exception(f"Kaggle CLI command failed with exit code {result}")
                
                # Verify download
                zip_path = f"{competition_name}.zip"
                if not os.path.exists(zip_path):
                    raise Exception("Downloaded zip file not found")
                
                logger.info(f"Kaggle download completed: {zip_path}")
                return True
                
            finally:
                os.chdir(original_directory)
                
        except Exception as e:
            logger.error(f"Kaggle download execution failed: {e}")
            return False
    
    def _fallback_zip_extraction(self, directory: str, zip_filename: str) -> bool:
        """Fallback zip extraction without FileSystemUtilities."""
        try:
            zip_path = os.path.join(directory, zip_filename)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(directory)
            logger.info(f"Extracted {zip_filename} using fallback method")
            return True
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return False


# Export all classes for external use
__all__ = [
    'KaggleManager'
]