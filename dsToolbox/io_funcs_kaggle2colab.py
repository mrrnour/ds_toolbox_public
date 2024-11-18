
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ----------------------------copy files from kaggle------------------
def copy_kaggle_json_to_colab(kaggle_json_source):
    """
    Copies the Kaggle JSON file containing API credentials to the appropriate location on Colab.

    Args:
        kaggle_json_source (str): The path to the 'kaggle.json' file on your system.
    """
    import os
    import shutil
    from google.colab import drive
    drive.mount('/content/drive/')


    kaggle_json_dest = os.path.expanduser('~/.kaggle')
    if not os.path.exists(kaggle_json_dest):
        os.mkdir(kaggle_json_dest)
    shutil.copyfile(kaggle_json_source, os.path.join(kaggle_json_dest, 'kaggle.json'))
    os.chmod(kaggle_json_dest, 0o600)  # Set permissions

def download_and_extract_dataset(download_folder, zip_file_name, extract_folders=None, exclude_folders=None):
    """
    Downloads and extracts the Kaggle dataset.

    Args:
        download_folder (str): The path to the directory where the dataset will be downloaded and extracted.
        zip_file_name (str): The name of the downloaded zip file.
        extract_folders (tuple, optional): A tuple of folders to extract from the zip file. If None, all folders are extracted. Defaults to None.
        exclude_folders (tuple, optional): A tuple of folders to exclude from extraction. If None, no folders are excluded. Defaults to None.
    """
    import zipfile
    import os
    from tqdm import tqdm
    from google.colab import drive
    drive.mount('/content/drive/')

    ##Download the dataset
    print("Download the dataset...")
    if not os.path.exists(download_folder):
        os.mkdir(download_folder)
    os.chdir(download_folder)
    os.system(f"kaggle competitions download -c {zip_file_name.split('.')[0]}")

    ## Extract specific folders
    print('unzip files...')
    zip_file_path = os.path.join(download_folder, zip_file_name)
    with zipfile.ZipFile(zip_file_path, 'r') as archive:
        files_to_extract = archive.namelist()

        ## Apply filtering based on extract_folders and exclude_folders
        if (extract_folders):
            files_to_extract = [file for file in files_to_extract if file.startswith(extract_folders)]
        if (exclude_folders):
            files_to_extract = [file for file in files_to_extract if not file.startswith(exclude_folders)]

        for file in tqdm(files_to_extract, desc="Extracting files"):
            archive.extract(file, download_folder)
