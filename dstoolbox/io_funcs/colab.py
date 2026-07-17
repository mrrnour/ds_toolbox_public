def setup_github_colab(user_email, user_name, ssh_source='/content/drive/My Drive/Colab Notebooks/.ssh'):
    """Mount Google Drive and configure SSH + git identity for GitHub access in Colab.

    Copies the user's SSH keys from a Drive folder to ``~/.ssh`` (only
    if missing), sets ``git config --global`` user email/name, and
    pings GitHub via SSH to verify.

    Parameters
    ----------
    user_email : str
        Value for ``git config --global user.email``.
    user_name : str
        Value for ``git config --global user.name``.
    ssh_source : str, optional
        Drive path holding the ``.ssh`` directory to copy. Defaults to
        ``'/content/drive/My Drive/Colab Notebooks/.ssh'``.

    Returns
    -------
    None
    """
    import os
    import shutil
    from google.colab import drive
    drive.mount('/content/drive/')
    if not os.path.exists(os.path.expanduser('~/.ssh')):
        shutil.copytree(ssh_source, os.path.expanduser('~/.ssh'))
    else: 
        "folder .ssh exists!"
    os.system(f'git config --global user.email "{user_email}"')
    os.system(f'git config --global user.name "{user_name}"')
    os.system('ssh -T git@github.com')

def copy_kaggle_json_to_colab(kaggle_json_source):
    """Copy a Kaggle ``kaggle.json`` credentials file into ``~/.kaggle`` on Colab.

    Mounts Google Drive, ensures ``~/.kaggle`` exists, copies the file,
    and sets restrictive permissions.

    Parameters
    ----------
    kaggle_json_source : str
        Path to the ``kaggle.json`` file to copy.
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
    """Download a Kaggle competition dataset and extract selected folders from its zip.

    Mounts Google Drive, downloads the zip via the ``kaggle`` CLI, then
    extracts entries whose paths pass the optional include/exclude filters.

    Parameters
    ----------
    download_folder : str
        Local directory to download and extract into (created if missing).
    zip_file_name : str
        The name of the downloaded zip file; the competition slug is taken
        from ``zip_file_name.split('.')[0]``.
    extract_folders : tuple of str, optional
        If given, only zip entries whose path starts with one of these
        prefixes are extracted.
    exclude_folders : tuple of str, optional
        If given, zip entries whose path starts with one of these
        prefixes are skipped.
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
