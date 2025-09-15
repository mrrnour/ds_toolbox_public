"""
Google Colab I/O Functions - Pragmatic Refactored Version
=========================================================

This module provides utilities for working with Google Colab including GitHub setup,
Kaggle integration, and dataset management with a balanced approach that groups
related functionality without over-engineering.

Classes:
    - ColabEnvironmentManager: Handles environment setup, Git configuration, and credentials
    - ColabDataManager: Manages dataset downloads and file operations

Utility Functions:
    - Drive mounting utilities
    - File system operations
    - Archive extraction helpers

Author: Refactored from original procedural code
"""

import os
import shutil
import zipfile
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

try:
    from google.colab import drive
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False
    warnings.warn("Google Colab not available. Some functionality will be limited.")


class ColabEnvironmentManager:
    """
    Manages Google Colab environment setup including Git configuration and credentials.
    
    This class handles Git/GitHub setup, SSH key management, and credential configuration
    for various services like Kaggle within the Google Colab environment.
    
    Attributes:
        drive_mounted (bool): Whether Google Drive has been mounted
        ssh_configured (bool): Whether SSH keys have been configured
        git_configured (bool): Whether Git has been configured
        
    Example:
        >>> env_mgr = ColabEnvironmentManager()
        >>> env_mgr.setup_github_integration('user@example.com', 'John Doe')
        >>> env_mgr.setup_kaggle_credentials('/content/drive/path/kaggle.json')
    """
    
    def __init__(self):
        """Initialize the Colab environment manager."""
        if not COLAB_AVAILABLE:
            warnings.warn("Google Colab environment not detected. Some features may not work.")
        
        self.drive_mounted = False
        self.ssh_configured = False
        self.git_configured = False
        self._drive_mount_point = '/content/drive'
    
    def mount_google_drive(self, mount_point: str = '/content/drive', force_remount: bool = False) -> bool:
        """
        Mount Google Drive in the Colab environment.
        
        This method mounts Google Drive to make files accessible within the Colab runtime.
        It includes error handling and prevents duplicate mounting.
        
        Parameters:
            mount_point (str): Directory where Drive should be mounted
            force_remount (bool): Whether to force remounting if already mounted
            
        Returns:
            bool: True if mounting successful, False otherwise
            
        Raises:
            RuntimeError: If not running in Google Colab environment
            Exception: If mounting fails
            
        Example:
            >>> env_mgr = ColabEnvironmentManager()
            >>> success = env_mgr.mount_google_drive()
            >>> if success:
            ...     print("Drive mounted successfully")
        """
        if not COLAB_AVAILABLE:
            raise RuntimeError("Google Drive mounting requires Google Colab environment")
        
        try:
            # Check if already mounted
            if os.path.ismount(mount_point) and not force_remount:
                print(f"Google Drive already mounted at {mount_point}")
                self.drive_mounted = True
                self._drive_mount_point = mount_point
                return True
            
            # Mount drive
            print(f"Mounting Google Drive at {mount_point}...")
            drive.mount(mount_point)
            
            # Verify mount was successful
            if os.path.exists(mount_point) and os.listdir(mount_point):
                self.drive_mounted = True
                self._drive_mount_point = mount_point
                print("Google Drive mounted successfully")
                return True
            else:
                warnings.warn("Drive mounted but appears to be empty")
                return False
                
        except Exception as e:
            error_msg = f"Failed to mount Google Drive: {str(e)}"
            warnings.warn(error_msg)
            return False
    
    def setup_github_integration(self, user_email: str, user_name: str,
                                ssh_source_path: str = '/content/drive/My Drive/Colab Notebooks/.ssh',
                                verify_connection: bool = True) -> bool:
        """
        Set up GitHub integration with SSH keys and Git configuration.
        
        This method configures Git with user credentials, copies SSH keys from Drive,
        and optionally verifies the GitHub connection.
        
        Parameters:
            user_email (str): Git user email address
            user_name (str): Git user name
            ssh_source_path (str): Path to SSH keys directory in Google Drive
            verify_connection (bool): Whether to test GitHub connection
            
        Returns:
            bool: True if setup successful, False otherwise
            
        Raises:
            ValueError: If email or name is empty
            RuntimeError: If not in Colab environment
            Exception: If setup fails
            
        Example:
            >>> env_mgr = ColabEnvironmentManager()
            >>> success = env_mgr.setup_github_integration(
            ...     'developer@company.com',
            ...     'Jane Developer',
            ...     ssh_source_path='/content/drive/My Drive/.ssh'
            ... )
        """
        if not user_email or not user_email.strip():
            raise ValueError("User email cannot be empty")
        
        if not user_name or not user_name.strip():
            raise ValueError("User name cannot be empty")
        
        if not COLAB_AVAILABLE:
            raise RuntimeError("GitHub integration requires Google Colab environment")
        
        try:
            # Ensure Drive is mounted
            if not self.drive_mounted:
                if not self.mount_google_drive():
                    raise Exception("Failed to mount Google Drive")
            
            # Validate SSH source path
            if not os.path.exists(ssh_source_path):
                raise FileNotFoundError(f"SSH source directory not found: {ssh_source_path}")
            
            # Copy SSH keys
            success = self._setup_ssh_keys(ssh_source_path)
            if not success:
                return False
            
            # Configure Git
            success = self._configure_git(user_email, user_name)
            if not success:
                return False
            
            # Verify GitHub connection
            if verify_connection:
                success = self._verify_github_connection()
                if not success:
                    warnings.warn("GitHub connection verification failed")
                    # Don't return False here as setup might still be valid
            
            print("GitHub integration setup completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"GitHub integration setup failed: {str(e)}"
            warnings.warn(error_msg)
            return False
    
    def _setup_ssh_keys(self, ssh_source_path: str) -> bool:
        """Set up SSH keys by copying from Google Drive."""
        try:
            ssh_dest_path = os.path.expanduser('~/.ssh')
            
            if os.path.exists(ssh_dest_path):
                if os.listdir(ssh_dest_path):
                    print("SSH directory already exists and is not empty")
                    self.ssh_configured = True
                    return True
                else:
                    # Remove empty SSH directory
                    os.rmdir(ssh_dest_path)
            
            # Copy SSH directory
            print(f"Copying SSH keys from {ssh_source_path} to {ssh_dest_path}")
            shutil.copytree(ssh_source_path, ssh_dest_path)
            
            # Set proper permissions
            os.chmod(ssh_dest_path, 0o700)
            
            # Set permissions for key files
            for key_file in ['id_rsa', 'id_ed25519', 'id_ecdsa']:
                key_path = os.path.join(ssh_dest_path, key_file)
                if os.path.exists(key_path):
                    os.chmod(key_path, 0o600)
            
            self.ssh_configured = True
            print("SSH keys configured successfully")
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to setup SSH keys: {str(e)}")
            return False
    
    def _configure_git(self, user_email: str, user_name: str) -> bool:
        """Configure Git with user credentials."""
        try:
            # Configure Git user email
            result = os.system(f'git config --global user.email "{user_email}"')
            if result != 0:
                raise Exception("Failed to set Git user email")
            
            # Configure Git user name
            result = os.system(f'git config --global user.name "{user_name}"')
            if result != 0:
                raise Exception("Failed to set Git user name")
            
            self.git_configured = True
            print(f"Git configured for user: {user_name} <{user_email}>")
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to configure Git: {str(e)}")
            return False
    
    def _verify_github_connection(self) -> bool:
        """Verify GitHub SSH connection."""
        try:
            print("Testing GitHub SSH connection...")
            result = os.system('ssh -T git@github.com -o ConnectTimeout=10 -o StrictHostKeyChecking=no')
            
            # SSH returns 1 for successful authentication with GitHub (by design)
            if result in [0, 256]:  # 256 = 1 * 256 (exit code 1)
                print("GitHub connection verified successfully")
                return True
            else:
                warnings.warn("GitHub connection test failed")
                return False
                
        except Exception as e:
            warnings.warn(f"GitHub connection verification failed: {str(e)}")
            return False
    
    def setup_kaggle_credentials(self, kaggle_json_source_path: str) -> bool:
        """
        Set up Kaggle API credentials in the Colab environment.
        
        This method copies the Kaggle API credentials file from Google Drive to the
        appropriate location and sets proper permissions for secure access.
        
        Parameters:
            kaggle_json_source_path (str): Path to kaggle.json file in Google Drive
            
        Returns:
            bool: True if setup successful, False otherwise
            
        Raises:
            ValueError: If source path is invalid
            FileNotFoundError: If kaggle.json file not found
            RuntimeError: If not in Colab environment
            
        Example:
            >>> env_mgr = ColabEnvironmentManager()
            >>> success = env_mgr.setup_kaggle_credentials(
            ...     '/content/drive/My Drive/credentials/kaggle.json'
            ... )
            >>> if success:
            ...     print("Kaggle credentials configured")
        """
        if not kaggle_json_source_path or not kaggle_json_source_path.strip():
            raise ValueError("Kaggle JSON source path cannot be empty")
        
        if not COLAB_AVAILABLE:
            raise RuntimeError("Kaggle setup requires Google Colab environment")
        
        try:
            # Ensure Drive is mounted
            if not self.drive_mounted:
                if not self.mount_google_drive():
                    raise Exception("Failed to mount Google Drive")
            
            # Validate source file
            if not os.path.exists(kaggle_json_source_path):
                raise FileNotFoundError(f"Kaggle JSON file not found: {kaggle_json_source_path}")
            
            if not kaggle_json_source_path.endswith('kaggle.json'):
                warnings.warn("Source file doesn't appear to be kaggle.json")
            
            # Create destination directory
            kaggle_dest_dir = os.path.expanduser('~/.kaggle')
            if not os.path.exists(kaggle_dest_dir):
                os.makedirs(kaggle_dest_dir, mode=0o700)
                print(f"Created Kaggle directory: {kaggle_dest_dir}")
            
            # Copy credentials file
            kaggle_dest_path = os.path.join(kaggle_dest_dir, 'kaggle.json')
            shutil.copyfile(kaggle_json_source_path, kaggle_dest_path)
            
            # Set secure permissions
            os.chmod(kaggle_dest_path, 0o600)
            
            # Verify the file was copied correctly
            if os.path.exists(kaggle_dest_path):
                print("Kaggle credentials configured successfully")
                
                # Test Kaggle API (optional)
                try:
                    test_result = os.system('kaggle --version > /dev/null 2>&1')
                    if test_result == 0:
                        print("Kaggle API is ready to use")
                    else:
                        warnings.warn("Kaggle API test failed. You may need to install kaggle: !pip install kaggle")
                except:
                    pass
                
                return True
            else:
                raise Exception("Failed to copy Kaggle credentials file")
                
        except Exception as e:
            error_msg = f"Kaggle credentials setup failed: {str(e)}"
            warnings.warn(error_msg)
            return False
    
    def get_environment_status(self) -> Dict[str, bool]:
        """
        Get the current status of Colab environment setup.
        
        Returns:
            Dict[str, bool]: Dictionary with setup status for each component
            
        Example:
            >>> env_mgr = ColabEnvironmentManager()
            >>> status = env_mgr.get_environment_status()
            >>> print(f"Drive mounted: {status['drive_mounted']}")
            >>> print(f"Git configured: {status['git_configured']}")
        """
        return {
            'colab_available': COLAB_AVAILABLE,
            'drive_mounted': self.drive_mounted,
            'ssh_configured': self.ssh_configured,
            'git_configured': self.git_configured,
            'kaggle_credentials': os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')),
        }


class ColabDataManager:
    """
    Manages data operations in Google Colab including dataset downloads and extractions.
    
    This class handles downloading datasets from Kaggle, extracting archives with
    filtering options, and managing file operations within the Colab environment.
    
    Example:
        >>> data_mgr = ColabDataManager()
        >>> success = data_mgr.download_kaggle_dataset(
        ...     'titanic', '/content/datasets', 
        ...     extract_folders=('train.csv', 'test.csv')
        ... )
    """
    
    def __init__(self):
        """Initialize the Colab data manager."""
        self.env_manager = ColabEnvironmentManager()
    
    def download_kaggle_dataset(self, competition_name: str, download_directory: str,
                               extract_folders: Optional[Tuple[str, ...]] = None,
                               exclude_folders: Optional[Tuple[str, ...]] = None,
                               clean_up_zip: bool = True,
                               create_directory: bool = True) -> bool:
        """
        Download and extract Kaggle competition dataset.
        
        This method downloads a Kaggle competition dataset, extracts it with optional
        filtering, and provides comprehensive error handling and progress tracking.
        
        Parameters:
            competition_name (str): Name of the Kaggle competition
            download_directory (str): Directory to download and extract files
            extract_folders (Optional[Tuple[str, ...]]): Specific folders/files to extract
            exclude_folders (Optional[Tuple[str, ...]]): Folders/files to exclude
            clean_up_zip (bool): Whether to delete zip file after extraction
            create_directory (bool): Whether to create download directory if it doesn't exist
            
        Returns:
            bool: True if download and extraction successful, False otherwise
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If Kaggle API not configured
            Exception: If download or extraction fails
            
        Example:
            >>> data_mgr = ColabDataManager()
            >>> 
            >>> # Download full dataset
            >>> success = data_mgr.download_kaggle_dataset('titanic', '/content/titanic_data')
            >>> 
            >>> # Download specific files only
            >>> success = data_mgr.download_kaggle_dataset(
            ...     'house-prices-advanced-regression-techniques',
            ...     '/content/housing_data',
            ...     extract_folders=('train.csv', 'test.csv', 'sample_submission.csv')
            ... )
            >>> 
            >>> # Exclude certain files
            >>> success = data_mgr.download_kaggle_dataset(
            ...     'nlp-getting-started',
            ...     '/content/nlp_data',
            ...     exclude_folders=('sample_submission.csv',)
            ... )
        """
        if not competition_name or not competition_name.strip():
            raise ValueError("Competition name cannot be empty")
        
        if not download_directory or not download_directory.strip():
            raise ValueError("Download directory cannot be empty")
        
        if not COLAB_AVAILABLE:
            warnings.warn("Not running in Google Colab environment")
        
        try:
            # Ensure Google Drive is mounted
            if not self.env_manager.drive_mounted:
                self.env_manager.mount_google_drive()
            
            # Validate Kaggle credentials
            if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
                raise RuntimeError(
                    "Kaggle credentials not found. Use setup_kaggle_credentials() first."
                )
            
            # Create download directory
            download_path = Path(download_directory)
            if create_directory:
                download_path.mkdir(parents=True, exist_ok=True)
            elif not download_path.exists():
                raise FileNotFoundError(f"Download directory does not exist: {download_directory}")
            
            # Download dataset
            success = self._download_kaggle_competition(competition_name, download_directory)
            if not success:
                return False
            
            # Extract dataset
            zip_filename = f"{competition_name}.zip"
            success = self._extract_dataset(
                download_directory, zip_filename, extract_folders, exclude_folders
            )
            if not success:
                return False
            
            # Clean up zip file if requested
            if clean_up_zip:
                zip_path = os.path.join(download_directory, zip_filename)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                    print(f"Cleaned up zip file: {zip_filename}")
            
            print(f"Dataset '{competition_name}' downloaded and extracted successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to download Kaggle dataset '{competition_name}': {str(e)}"
            warnings.warn(error_msg)
            return False
    
    def _download_kaggle_competition(self, competition_name: str, download_directory: str) -> bool:
        """Download Kaggle competition dataset."""
        try:
            print(f"Downloading Kaggle competition: {competition_name}")
            
            # Change to download directory
            original_cwd = os.getcwd()
            os.chdir(download_directory)
            
            try:
                # Execute Kaggle download command
                download_command = f"kaggle competitions download -c {competition_name}"
                result = os.system(download_command)
                
                if result != 0:
                    raise Exception(f"Kaggle download command failed with exit code {result}")
                
                # Verify download
                zip_path = os.path.join(download_directory, f"{competition_name}.zip")
                if not os.path.exists(zip_path):
                    raise Exception("Downloaded zip file not found")
                
                print(f"Download completed: {os.path.basename(zip_path)}")
                return True
                
            finally:
                # Always return to original directory
                os.chdir(original_cwd)
                
        except Exception as e:
            warnings.warn(f"Kaggle download failed: {str(e)}")
            return False
    
    def _extract_dataset(self, download_directory: str, zip_filename: str,
                        extract_folders: Optional[Tuple[str, ...]] = None,
                        exclude_folders: Optional[Tuple[str, ...]] = None) -> bool:
        """Extract dataset with filtering options."""
        try:
            from tqdm import tqdm
            
            zip_path = os.path.join(download_directory, zip_filename)
            
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"Zip file not found: {zip_path}")
            
            print("Extracting files...")
            
            with zipfile.ZipFile(zip_path, 'r') as archive:
                # Get list of all files in archive
                all_files = archive.namelist()
                
                if not all_files:
                    warnings.warn("Zip file appears to be empty")
                    return False
                
                # Apply filtering
                files_to_extract = self._filter_files_for_extraction(
                    all_files, extract_folders, exclude_folders
                )
                
                if not files_to_extract:
                    warnings.warn("No files selected for extraction after filtering")
                    return False
                
                # Extract filtered files
                print(f"Extracting {len(files_to_extract)} files...")
                
                for file_name in tqdm(files_to_extract, desc="Extracting files"):
                    try:
                        archive.extract(file_name, download_directory)
                    except Exception as e:
                        warnings.warn(f"Failed to extract {file_name}: {str(e)}")
                        continue
                
                print(f"Extraction completed: {len(files_to_extract)} files extracted")
                return True
                
        except Exception as e:
            warnings.warn(f"Dataset extraction failed: {str(e)}")
            return False
    
    def _filter_files_for_extraction(self, all_files: List[str],
                                    extract_folders: Optional[Tuple[str, ...]] = None,
                                    exclude_folders: Optional[Tuple[str, ...]] = None) -> List[str]:
        """Filter files based on extraction and exclusion criteria."""
        files_to_extract = all_files.copy()
        
        # Apply inclusion filter
        if extract_folders:
            files_to_extract = [
                file for file in files_to_extract
                if any(file.startswith(folder) for folder in extract_folders)
            ]
            print(f"Filtered to include only: {extract_folders}")
        
        # Apply exclusion filter  
        if exclude_folders:
            files_to_extract = [
                file for file in files_to_extract
                if not any(file.startswith(folder) for folder in exclude_folders)
            ]
            print(f"Excluded: {exclude_folders}")
        
        return files_to_extract
    
    def extract_archive(self, archive_path: str, extract_directory: str,
                       extract_folders: Optional[Tuple[str, ...]] = None,
                       exclude_folders: Optional[Tuple[str, ...]] = None) -> bool:
        """
        Extract archive file with filtering options.
        
        This method provides a general-purpose archive extraction utility that can
        be used for any zip file, not just Kaggle datasets.
        
        Parameters:
            archive_path (str): Path to the archive file
            extract_directory (str): Directory to extract files to
            extract_folders (Optional[Tuple[str, ...]]): Specific folders/files to extract
            exclude_folders (Optional[Tuple[str, ...]]): Folders/files to exclude
            
        Returns:
            bool: True if extraction successful, False otherwise
            
        Example:
            >>> data_mgr = ColabDataManager()
            >>> success = data_mgr.extract_archive(
            ...     '/content/my_data.zip',
            ...     '/content/extracted',
            ...     extract_folders=('important_files/',)
            ... )
        """
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        
        # Create extraction directory
        Path(extract_directory).mkdir(parents=True, exist_ok=True)
        
        # Extract using the same logic as dataset extraction
        archive_name = os.path.basename(archive_path)
        success = self._extract_dataset(
            os.path.dirname(archive_path), archive_name, extract_folders, exclude_folders
        )
        
        return success
    
    def list_archive_contents(self, archive_path: str) -> List[str]:
        """
        List contents of an archive file.
        
        Parameters:
            archive_path (str): Path to the archive file
            
        Returns:
            List[str]: List of file names in the archive
            
        Example:
            >>> data_mgr = ColabDataManager()
            >>> files = data_mgr.list_archive_contents('/content/dataset.zip')
            >>> for file in files:
            ...     print(file)
        """
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        
        try:
            with zipfile.ZipFile(archive_path, 'r') as archive:
                return archive.namelist()
        except Exception as e:
            raise Exception(f"Failed to list archive contents: {str(e)}")


# ============================================================================
# UTILITY FUNCTIONS (Simple functions for common operations)
# ============================================================================

def mount_google_drive_simple(mount_point: str = '/content/drive') -> bool:
    """
    Simple utility to mount Google Drive.
    
    Parameters:
        mount_point (str): Directory where Drive should be mounted
        
    Returns:
        bool: True if mounting successful, False otherwise
        
    Example:
        >>> success = mount_google_drive_simple()
        >>> if success:
        ...     print("Drive is ready to use")
    """
    env_manager = ColabEnvironmentManager()
    return env_manager.mount_google_drive(mount_point)


def check_colab_environment() -> Dict[str, Any]:
    """
    Check the current Google Colab environment status.
    
    Returns:
        Dict[str, Any]: Environment status information
        
    Example:
        >>> status = check_colab_environment()
        >>> print(f"Running in Colab: {status['colab_available']}")
        >>> print(f"Drive mounted: {status['drive_mounted']}")
    """
    env_manager = ColabEnvironmentManager()
    status = env_manager.get_environment_status()
    
    # Add additional environment info
    status.update({
        'current_directory': os.getcwd(),
        'python_version': os.sys.version,
        'drive_contents_accessible': os.path.exists('/content/drive/My Drive') if status['drive_mounted'] else False
    })
    
    return status


def setup_colab_workspace(user_email: str, user_name: str,
                         ssh_source: str = '/content/drive/My Drive/Colab Notebooks/.ssh',
                         kaggle_json_source: Optional[str] = None) -> Dict[str, bool]:
    """
    Complete workspace setup for Google Colab.
    
    This utility function performs a full workspace setup including drive mounting,
    Git configuration, SSH setup, and optionally Kaggle credentials.
    
    Parameters:
        user_email (str): Git user email
        user_name (str): Git user name  
        ssh_source (str): Path to SSH keys in Google Drive
        kaggle_json_source (Optional[str]): Path to Kaggle JSON credentials
        
    Returns:
        Dict[str, bool]: Setup results for each component
        
    Example:
        >>> results = setup_colab_workspace(
        ...     'user@example.com', 'John Doe',
        ...     kaggle_json_source='/content/drive/My Drive/kaggle.json'
        ... )
        >>> print(f"GitHub setup: {results['github_setup']}")
        >>> print(f"Kaggle setup: {results['kaggle_setup']}")
    """
    env_manager = ColabEnvironmentManager()
    
    results = {
        'drive_mounted': False,
        'github_setup': False,
        'kaggle_setup': False
    }
    
    try:
        # Mount Google Drive
        results['drive_mounted'] = env_manager.mount_google_drive()
        
        # Setup GitHub integration
        if results['drive_mounted']:
            results['github_setup'] = env_manager.setup_github_integration(
                user_email, user_name, ssh_source
            )
        
        # Setup Kaggle credentials if provided
        if kaggle_json_source and results['drive_mounted']:
            results['kaggle_setup'] = env_manager.setup_kaggle_credentials(kaggle_json_source)
        else:
            results['kaggle_setup'] = None  # Not attempted
        
        # Print summary
        print("\n" + "="*50)
        print("COLAB WORKSPACE SETUP SUMMARY")
        print("="*50)
        print(f"Drive mounted: {'✓' if results['drive_mounted'] else '✗'}")
        print(f"GitHub setup: {'✓' if results['github_setup'] else '✗'}")
        if results['kaggle_setup'] is not None:
            print(f"Kaggle setup: {'✓' if results['kaggle_setup'] else '✗'}")
        print("="*50)
        
    except Exception as e:
        warnings.warn(f"Workspace setup encountered errors: {str(e)}")
    
    return results


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def setup_github_colab(user_email, user_name, ssh_source='/content/drive/My Drive/Colab Notebooks/.ssh'):
    """Legacy function for backward compatibility."""
    env_manager = ColabEnvironmentManager()
    return env_manager.setup_github_integration(user_email, user_name, ssh_source)


def copy_kaggle_json_to_colab(kaggle_json_source):
    """Legacy function for backward compatibility."""
    env_manager = ColabEnvironmentManager()
    return env_manager.setup_kaggle_credentials(kaggle_json_source)


def download_and_extract_dataset(download_folder, zip_file_name, extract_folders=None, exclude_folders=None):
    """Legacy function for backward compatibility."""
    data_manager = ColabDataManager()
    
    # Extract competition name from zip file name
    competition_name = zip_file_name.split('.')[0] if '.' in zip_file_name else zip_file_name
    
    return data_manager.download_kaggle_dataset(
        competition_name, download_folder, extract_folders, exclude_folders
    )


# Function mapping for reference
FUNCTION_MAPPING = {
    'setup_github_colab': 'ColabEnvironmentManager.setup_github_integration()',
    'copy_kaggle_json_to_colab': 'ColabEnvironmentManager.setup_kaggle_credentials()',
    'download_and_extract_dataset': 'ColabDataManager.download_kaggle_dataset()',
}


def print_function_mapping():
    """Print the mapping of old functions to new implementations."""
    print("Function Mapping - Old to New:")
    print("=" * 70)
    for old_func, new_impl in FUNCTION_MAPPING.items():
        print(f"{old_func:30} -> {new_impl}")
    print("=" * 70)
    
    print("\nAdditional Utilities:")
    print("- mount_google_drive_simple()")
    print("- check_colab_environment()")
    print("- setup_colab_workspace()")


# Example usage
if __name__ == "__main__":
    # Print function mapping for reference
    print_function_mapping()
    
    # Example usage
    print("\nExample Usage:")
    print("# Object-oriented approach:")
    print("env_mgr = ColabEnvironmentManager()")
    print("env_mgr.setup_github_integration('user@email.com', 'User Name')")
    print("data_mgr = ColabDataManager()")
    print("data_mgr.download_kaggle_dataset('titanic', '/content/data')")
    
    print("\n# Backward compatible approach:")
    print("setup_github_colab('user@email.com', 'User Name')")
    print("copy_kaggle_json_to_colab('/path/to/kaggle.json')")
    print("download_and_extract_dataset('/content/data', 'competition.zip')")
    
    print("\n# New utility functions:")
    print("setup_colab_workspace('user@email.com', 'User Name', kaggle_json_source='/path/kaggle.json')")
    print("status = check_colab_environment()")