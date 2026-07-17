"""Azure Blob storage I/O: spark/pandas read & write, batch upload, existence check, multi-sheet xlsx.

All entry points take ``blob_dict = {'target_id': ..., 'container': ..., 'blob': ...}``
where ``target_id`` is a key under ``data_sources:`` in config.yml.
"""

import io
import logging
import os
import inspect
import pandas as pd

from .bootstrap import get_spark
from . import data_sources
from .exceptions import BlobError

logger_mod = logging.getLogger(__name__)


_TMP_FILE_LOCS = {
    "databricks": "/tmp",
    "aml": None,  # use cwd
    "local": None,
    "vm_docker": None,
}


def _tmp_path(runtime: str, blob: str) -> str:
    base = _TMP_FILE_LOCS.get(runtime)
    if base is None:
        base = os.getcwd()
    return os.path.join(base, os.path.basename(blob))


def _extension(blob: str) -> str:
    return blob.rsplit(".", 1)[-1].lower()


def _resolve_blob(blob_dict: dict, runtime: str):
    """Return ``(ds, container, blob)`` for a blob_dict + runtime."""
    target_id = blob_dict["target_id"]
    container = blob_dict["container"]
    blob = blob_dict["blob"]
    ds = data_sources.get(target_id, runtime=runtime)
    return ds, container, blob


def blob2spark(blob_dict: dict, runtime: str = "databricks"):
    """Read a blob file (csv or parquet) as a Spark DataFrame.

    Parameters
    ----------
    blob_dict : dict
        Must contain ``target_id``, ``container``, ``blob``.
    runtime : str, default ``'databricks'``
        Passed to ``data_sources.get`` to select the credential path.

    Returns
    -------
    pyspark.sql.DataFrame
        The loaded frame.

    Raises
    ------
    ValueError
        If the file extension is not ``.csv`` or ``.parquet``.
    """
    ds, container, blob = _resolve_blob(blob_dict, runtime)
    spark, _ = get_spark()
    spark.conf.set(ds.spark_conf_key, ds.account_key)
    uri = ds.wasbs_uri(container, blob)
    ext = _extension(blob)
    if ext == "csv":
        return (
            spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load(uri)
        )
    if ext == "parquet":
        return spark.read.format("parquet").load(uri)
    raise ValueError(f"blob2spark: unsupported extension {ext!r}")


def spark2blob(df, blob_dict: dict, write_mode: str = "append",
               runtime: str = "databricks"):
    """Save a Spark DataFrame to Azure blob storage.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Data to write. Format is inferred from the blob file extension.
    blob_dict : dict
        Must contain ``target_id``, ``container``, ``blob``.
    write_mode : str, default ``'append'``
        Passed directly to ``DataFrameWriter.mode`` (e.g. ``'append'``,
        ``'overwrite'``, ``'error'``, ``'ignore'``).
    runtime : str, default ``'databricks'``
        Passed to ``data_sources.get``.
    """
    ds, container, blob = _resolve_blob(blob_dict, runtime)
    spark, _ = get_spark()
    spark.conf.set(ds.spark_conf_key, ds.account_key)
    uri = ds.wasbs_uri(container, blob)
    ext = _extension(blob)
    df.write.format(ext).mode(write_mode).save(uri)


def blob2pd(blob_dict: dict, verbose: bool = True, runtime: str = "databricks",
            load_to_memory: bool = False, **kwargs) -> pd.DataFrame:
    """Read a blob file (csv or parquet) as a pandas DataFrame.

    Parameters
    ----------
    blob_dict : dict
        Must contain ``target_id``, ``container``, ``blob``.
    verbose : bool, default True
        Print a one-line download banner.
    runtime : str, default ``'databricks'``
        Chooses temp-file location and credential path.
    load_to_memory : bool, default False
        If True, stream the blob into an in-memory buffer instead of
        writing to a temp file on disk.
    **kwargs
        Extra keyword args. Names matching ``pd.read_csv`` /
        ``pd.read_parquet`` are forwarded to the appropriate reader; the
        rest are ignored.

    Returns
    -------
    pandas.DataFrame or io.BytesIO or str
        A DataFrame for csv/parquet blobs; the raw buffer (if
        ``load_to_memory=True`` and the format isn't recognized); or the
        local path to the temp file (otherwise).
    """
    from azure.storage.blob import BlobServiceClient

    csv_args = list(inspect.signature(pd.read_csv).parameters)
    kwargs_csv = {k: kwargs.pop(k) for k in dict(kwargs) if k in csv_args}
    parq_args = list(inspect.signature(pd.read_parquet).parameters)
    kwargs_parq = {k: kwargs.pop(k) for k in dict(kwargs) if k in parq_args}

    ds, container, blob = _resolve_blob(blob_dict, runtime)
    bsc = BlobServiceClient.from_connection_string(ds.account_connection_string)
    blob_client = bsc.get_blob_client(container=container, blob=blob)
    ext = _extension(blob)

    if verbose:
        print(
            f"Downloading from target_id:{ds.target_id!r}, "
            f"container:{container!r}, blob:{blob!r}"
        )

    if load_to_memory:
        with io.BytesIO() as buf:
            blob_client.download_blob().readinto(buf)
            buf.seek(0)
            if ext == "csv":
                return pd.read_csv(buf, **kwargs_csv)
            if ext == "parquet":
                return pd.read_parquet(buf, **kwargs_parq)
            print("file uploaded into the memory")
            return buf

    tmp_file = _tmp_path(runtime, blob)
    with open(tmp_file, "wb") as dest:
        dest.write(blob_client.download_blob().readall())
    try:
        if ext == "csv":
            return pd.read_csv(tmp_file, **kwargs_csv)
        if ext == "parquet":
            return pd.read_parquet(tmp_file, **kwargs_parq)
        print(f"file uploaded in {tmp_file}")
        return tmp_file
    finally:
        if ext in ("csv", "parquet") and os.path.exists(tmp_file):
            os.remove(tmp_file)


def pd2blob(data: pd.DataFrame, blob_dict: dict, append: bool = False,
            overwrite: bool = True, runtime: str = "databricks",
            sheetName: str = "dataframe1", **kwargs):
    """Save a pandas DataFrame into Azure blob storage.

    Format is inferred from the blob file extension. Supported:
    ``csv``, ``parquet``, ``xlsx``; ``xls`` is rejected. When the target
    already exists and ``append == overwrite``, both flags are forced to
    ``append=True, overwrite=False`` and a note is printed.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to write.
    blob_dict : dict
        Must contain ``target_id``, ``container``, ``blob``.
    append : bool, default False
        For ``csv``/``parquet``, append to any existing blob.
    overwrite : bool, default True
        Overwrite an existing blob (ignored when appending).
    runtime : str, default ``'databricks'``
        Passed to ``data_sources.get``.
    sheetName : str, default ``'dataframe1'``
        Sheet name used when the blob has an ``.xlsx`` extension.
    **kwargs
        Extra kwargs. Names matching ``to_csv`` / ``to_parquet`` /
        ``to_excel`` / ``upload_blob`` are forwarded to the appropriate
        method; the rest are ignored.

    Returns
    -------
    azure.storage.blob.BlobProperties
        Properties of the uploaded blob.

    Raises
    ------
    BlobError
        For unsupported ``.xls`` extension or when appending to xlsx.
    """
    from azure.storage.blob import BlobServiceClient

    ds, container, blob = _resolve_blob(blob_dict, runtime)
    bsc = BlobServiceClient.from_connection_string(ds.account_connection_string)
    blob_client = bsc.get_container_client(container).get_blob_client(blob)

    if blob_client.exists() and append == overwrite:
        print(
            f"append and overwrite have value {append}; "
            "defaulting to append=True, overwrite=False"
        )
        append, overwrite = True, False

    csv_args = list(inspect.signature(pd.DataFrame.to_csv).parameters)
    kwargs_csv = {k: kwargs.pop(k) for k in dict(kwargs) if k in csv_args}
    parq_args = list(inspect.signature(pd.DataFrame.to_parquet).parameters)
    kwargs_parq = {k: kwargs.pop(k) for k in dict(kwargs) if k in parq_args}
    xls_args = list(inspect.signature(pd.DataFrame.to_excel).parameters)
    kwargs_xls = {k: kwargs.pop(k) for k in dict(kwargs) if k in xls_args}
    blob_args = list(inspect.signature(blob_client.upload_blob).parameters)
    kwargs_blob = {k: kwargs.pop(k) for k in dict(kwargs) if k in blob_args}

    ext = _extension(blob)
    if ext == "csv":
        if blob_client.exists() and append:
            blob_client.upload_blob(
                data=data.to_csv(header=False, **kwargs_csv),
                **kwargs_blob, blob_type="AppendBlob",
            )
        else:
            blob_client.upload_blob(
                data=data.to_csv(**kwargs_csv),
                **kwargs_blob, overwrite=overwrite, blob_type="AppendBlob",
            )
    elif ext == "parquet":
        if blob_client.exists() and append:
            df_current = blob2pd(blob_dict, runtime=runtime)
            df_current = pd.concat([df_current, data], axis=0)
            blob_client.upload_blob(
                data=df_current.to_parquet(**kwargs_parq),
                overwrite=True, **kwargs_blob,
            )
        else:
            blob_client.upload_blob(
                data=data.to_parquet(**kwargs_parq),
                overwrite=overwrite, **kwargs_blob,
            )
    elif ext == "xls":
        raise BlobError(
            "pd2blob does not support the legacy .xls format; use xlsx"
        )
    elif ext == "xlsx":
        if append and not overwrite:
            raise BlobError(
                "pd2blob does not append to existing xlsx; "
                "use xls2blob for multi-sheet writes"
            )
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            data.to_excel(writer, sheet_name=sheetName, **kwargs_xls)
        blob_client.upload_blob(
            data=out.getvalue(), overwrite=True, **kwargs_blob,
        )
    else:
        print("Append option is not usable for this extension")
        blob_client.upload_blob(
            data=data, overwrite=overwrite, **kwargs_blob,
        )

    return blob_client.get_blob_properties()


def pd2blob_batch(outputs: dict, blob_dict: dict, append: bool = True,
                  runtime: str = "databricks", **kwargs):
    """Save multiple pandas DataFrames into Azure blob storage.

    ``outputs`` maps blob name → DataFrame. Each entry is written via
    :func:`pd2blob` with the same ``blob_dict`` (its ``blob`` key is
    overwritten per iteration).
    """
    for out_blob, df in outputs.items():
        try:
            blob_dict = {**blob_dict, "blob": out_blob}
            pd2blob(df, blob_dict=blob_dict, runtime=runtime,
                    append=append, **kwargs)
            logger_mod.info("%s saved", out_blob)
        except Exception as e:
            logger_mod.error("Writing %s failed: %s", out_blob, e)


def blob_check(blob_dict: dict, runtime: str = "databricks") -> bool:
    """Return True iff the blob exists."""
    from azure.storage.blob import BlobClient

    ds, container, blob = _resolve_blob(blob_dict, runtime)
    client = BlobClient.from_connection_string(
        conn_str=ds.account_connection_string,
        container_name=container,
        blob_name=blob,
    )
    return client.exists()


def xls2blob(dataframe_dict: dict, blob_dict: dict, overwrite: bool = True,
             runtime: str = "databricks", **kwargs):
    """Save multiple pandas DataFrames as sheets of one xlsx in Azure blob storage."""
    from azure.storage.blob import BlobServiceClient

    ds, container, blob = _resolve_blob(blob_dict, runtime)
    bsc = BlobServiceClient.from_connection_string(ds.account_connection_string)
    blob_client = bsc.get_container_client(container).get_blob_client(blob)

    blob_args = list(inspect.signature(blob_client.upload_blob).parameters)
    kwargs_blob = {k: kwargs.pop(k) for k in dict(kwargs) if k in blob_args}
    xls_args = list(inspect.signature(pd.DataFrame.to_excel).parameters)
    kwargs_xls = {k: kwargs.pop(k) for k in dict(kwargs) if k in xls_args}

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        for sheet_name, df in dataframe_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, **kwargs_xls)
    blob_client.upload_blob(
        data=out.getvalue(), overwrite=overwrite, **kwargs_blob,
    )
    return blob_client.get_blob_properties()
