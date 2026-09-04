"""Filesystem and config helpers: path validation, output folder setup, yaml config + params loading."""

import argparse
import importlib.util
import os
import shutil

from .exceptions import InvalidConfigError, OutputFolderError


def check_path(path):
    """Raise exception if the file path doesn't exist."""
    if "~" in path:
        path = os.path.expanduser(path)
    if not os.path.exists(path):
        msg = f"File ({path}) not found!"
        raise argparse.ArgumentTypeError(msg)
    return path


def copy_ymls(dstoolbox, platform="databricks", destination=None):
    """Copy the bundled ``config.yml`` and ``sql_template.yml`` to a destination dir.

    Parameters
    ----------
    dstoolbox : module
        The ``dstoolbox`` package whose installed location holds the source
        YAMLs.
    platform : str, optional
        Currently only ``'databricks'`` triggers an actual copy (uses
        ``dbutils.fs.cp``); other platforms are no-ops here. Default
        ``'databricks'``.
    destination : str or None, optional
        Target directory. Defaults to the current working directory.

    Returns
    -------
    None
        Files are written for side effects.
    """
    ##TODO: add comments:
    from io_funcs import io_funcs

    upath = dstoolbox.__file__
    if destination is None:
        destination = os.getcwd()
    for ufile in ["config.yml", "sql_template.yml"]:
        ufile_src = os.path.join(os.path.dirname(upath), ufile)
        ufile_desc = os.path.join(destination, ufile)
        ufile_desc_tmp = os.path.join(destination, f".{ufile}.crc")
        print(f"copying {ufile_src} ---> {ufile_desc}")
        if platform == "databricks":
            dbutils = io_funcs.get_dbutils()
            dbutils.fs.cp(f"file://{ufile_src}", f"file://{ufile_desc}")
            dbutils.fs.rm(f"file://{ufile_desc_tmp}")


def setup_output_folder(outputFolder, uFiles, overWrite):
    """Create ``outputFolder`` if missing and optionally copy template files into it.

    Parameters
    ----------
    outputFolder : str
        Absolute or relative path of the output directory. Relative
        paths are resolved against ``os.getcwd()``.
    uFiles : list of str
        Files to copy into ``outputFolder`` (only when ``overWrite`` is
        True).
    overWrite : bool
        If True, always copy ``uFiles`` (creating the folder if needed).
        If False and the folder already exists, an ``OutputFolderError``
        is raised.

    Returns
    -------
    str
        Absolute path of the created/verified output folder.

    Raises
    ------
    OutputFolderError
        Folder exists and ``overWrite`` is False.
    """

    # Setting output directory and copying template file(s)
    if len(outputFolder.split("/")) == 1:
        outputFolder = os.path.abspath(os.path.join(os.getcwd(), outputFolder))
    else:
        outputFolder = os.path.abspath(outputFolder)

    if os.path.exists(outputFolder) & (not overWrite):
        raise OutputFolderError(
            "overwrite is not allowed and the output directory exists: " f"{outputFolder!r}"
        )
    # elif os.path.exists(outputFolder):
    #     shutil.rmtree(outputFolder)
    #     os.makedirs(outputFolder)
    # else:
    #     os.makedirs(outputFolder)
    elif not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if overWrite:
        for uFile in uFiles:
            shutil.copyfile(uFile, os.path.join(outputFolder, os.path.basename(uFile)))
    return outputFolder


def load_config(config_file, logger):
    """Load ``internet_credentials`` from a YAML config file.

    On any failure (file missing, invalid YAML, missing key), logs the
    error and returns empty-string credentials so callers can degrade
    gracefully.

    Parameters
    ----------
    config_file : str
        Path to the YAML config.
    logger : logging.Logger
        Logger used to report problems.

    Returns
    -------
    dict
        Mapping with keys ``username`` and ``password`` (empty strings on
        failure).
    """
    import yaml

    try:
        if not os.path.exists(config_file):
            logger.error(f"Config file {config_file} not found")
            return {"username": "", "password": ""}

        with open(config_file) as f:
            config = yaml.safe_load(f)

        if not config or "internet_credentials" not in config:
            logger.error("Invalid config file format: missing 'internet_credentials' section")
            return {"username": "", "password": ""}

        return config["internet_credentials"]
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_file}: {str(e)}")
        return {"username": "", "password": ""}
    except Exception as e:
        logger.error(f"Error loading {config_file}: {str(e)}")
        return {"username": "", "password": ""}


def load_params(param_file):
    """Import a Python file as a module and return it.

    Parameters
    ----------
    param_file : str
        Path to a ``.py`` file to load.

    Returns
    -------
    module
        The imported module object; attributes correspond to top-level
        names defined in the file.

    Raises
    ------
    InvalidConfigError
        The file cannot be imported.
    """
    try:
        spec = importlib.util.spec_from_file_location("params", param_file)
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)
        return params
    except Exception as e:
        raise InvalidConfigError(f"Error loading params file {param_file!r}: {e}") from e
