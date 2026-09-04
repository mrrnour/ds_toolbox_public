import datetime as dt

import pandas as pd

from dstoolbox import io_funcs, utils


def last_date(
    output_name,
    date_col="date",
    custom_config=None,
    key_vault_dict="deltaTable",  ###for only delta_table
    platform="databricks",  ### for only blob
):
    """Return the most recent date already stored in a Delta table or blob output.

    Parameters
    ----------
    output_name : str or dict
        ``str`` is treated as a fully-qualified Delta table name; a
        ``dict`` is treated as a blob descriptor (see ``io_funcs.blob2pd``).
    date_col : str, optional
        Name of the timestamp column in the output. Default ``'date'``.
    custom_config : dict or str, optional
        Forwarded to the underlying ``io_funcs`` call.
    key_vault_dict : str, optional
        Key Vault entry; only used for the Delta path. Default ``'deltaTable'``.
    platform : str, optional
        Platform tag for blob access. Default ``'databricks'``.

    Returns
    -------
    datetime-like or None
        The maximum value of ``date_col`` in the output, or ``None`` if
        the table/blob does not exist yet.
    """

    if (isinstance(output_name, str)) and (io_funcs.delta_table_check(output_name)):
        saved_dates = io_funcs.query_delta_table_db(
            f"""select min({date_col}) as min_time ,
                                                          max({date_col}) as max_time
                                                  from   {output_name}""",
            key_vault_dict=key_vault_dict,
            custom_config=custom_config,
            verbose=False,
        ).toPandas()
        last_save_date = saved_dates["max_time"].iloc[0]
        print(f"The last date found in delta_table:{last_save_date}")

    elif (isinstance(output_name, dict)) and (
        io_funcs.blob_check(
            blob_dict=output_name,
            custom_config=custom_config,
            platform=platform,
        )
    ):
        udata = io_funcs.blob2pd(
            blob_dict=output_name,
            custom_config=custom_config,
            platform=platform,
            #  **kwargs_csv,
        )
        udata[date_col] = pd.to_datetime(udata[date_col], format="%Y-%m-%d")
        last_save_date = udata[date_col].max()

        print(f"The last date found in blob:{last_save_date}")

    else:
        last_save_date = None

    return last_save_date


def save_outputs(
    outputs_dict_list,
    **kwargs,
):
    """Persist a batch of (target → frame) pairs to Delta tables or Azure blobs.

    Each item is dispatched based on its key type: ``str`` keys are
    written via ``io_funcs.spark2delta_table``; ``dict`` keys are blob
    descriptors written via ``io_funcs.pd2blob``.

    Parameters
    ----------
    outputs_dict_list : dict or list of (key, frame) pairs
        Mapping from output identifier to the (Spark or pandas) DataFrame
        to write.
    **kwargs
        Extra keyword args forwarded to either ``spark2delta_table`` or
        ``pd2blob`` based on which one accepts them.

    Returns
    -------
    None
    """
    import inspect

    from dstoolbox import io_funcs

    spark2del_args = list(inspect.signature(io_funcs.spark2delta_table).parameters)
    spark2del_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in spark2del_args}
    pd2blob_args = list(inspect.signature(io_funcs.pd2blob).parameters)
    pd2blob_args = {k: kwargs.pop(k) for k in dict(kwargs) if k in pd2blob_args}

    outputs = (
        outputs_dict_list.items() if isinstance(outputs_dict_list, dict) else outputs_dict_list
    )

    for tableName_blobDict, sp in outputs:
        print(tableName_blobDict)
        if isinstance(tableName_blobDict, str):
            io_funcs.spark2delta_table(
                sp,
                table_name=tableName_blobDict.split(".")[1],
                schema=tableName_blobDict.split(".")[0],
                write_mode="append",
                mergeSchema=True,
                **spark2del_args,
            )
        elif isinstance(tableName_blobDict, dict):
            io_funcs.pd2blob(
                sp, blob_dict=tableName_blobDict, overwrite=False, append=True, **pd2blob_args
            )


##TODO: update based on run_recursively
def update_db_recursively(
    df_generator_func,
    output_name,
    year_range=None,
    firstDate=None,
    lastDate=None,
    date_col="date",
    custom_config=None,
    key_vault_dict="deltaTable",  ###for  delta_table only
    platform="databricks",  ### for blob only
    **kwargs,
):
    """Run a generator function over each month-slice and append results to a Delta table or blob.

    Builds a list of month-start dates from ``year_range`` (or starting at
    the day after the last saved record if the output already exists),
    ending at ``lastDate``. For each consecutive pair of dates it calls
    ``df_generator_func(start_date, end_date, output_name, **kwargs)`` and
    hands the result(s) to :func:`save_outputs`.

    Example
    -------
    With ``year_range=[2021, 2099]``, last date in output ``'2023-07-15'``,
    and run date ``'2023-12-22'``, the generator runs once per month
    from ``2023-07-16`` through ``2023-12-22``.

    Parameters
    ----------
    df_generator_func : callable
        Function taking ``(start_date, end_date, output_name, **kwargs)``
        and returning the outputs in the form expected by
        :func:`save_outputs`.
    output_name : str or dict
        Delta table name (str) or blob descriptor (dict).
    year_range : list of int or None, optional
        ``[first_year, last_year]``. Defaults to ``[2021, 2099]``.
    firstDate : str, datetime.date, or None, optional
        Override the start of the date range. If ``None`` (default), the
        function continues from the last saved date in the output.
    lastDate : datetime.date or None, optional
        End of the date range. Defaults to today.
    date_col : str, optional
        Column used to find the last saved date. Default ``'date'``.
    custom_config : dict or str, optional
        Forwarded to the underlying ``io_funcs`` calls.
    key_vault_dict : str, optional
        Used only for Delta-table outputs. Default ``'deltaTable'``.
    platform : str, optional
        Used only for blob outputs. Default ``'databricks'``.
    **kwargs
        Anything else: arguments matching ``spark2delta_table`` or
        ``pd2blob`` are forwarded to those; remaining kwargs are passed to
        ``df_generator_func``.

    Returns
    -------
    None
        Writes are performed for side effects.
    """
    if year_range is None:
        year_range = [2021, 2099]
    if lastDate is None:
        lastDate = dt.datetime.now().date()
    import inspect

    from dstoolbox import io_funcs

    spark2del_args = list(inspect.signature(io_funcs.spark2delta_table).parameters)
    pd2blob_args = list(inspect.signature(io_funcs.pd2blob).parameters)
    df_generator_func_args = {
        k: kwargs.pop(k) for k in dict(kwargs) if k not in (spark2del_args + pd2blob_args)
    }

    import datetime as dt

    last_save_date = last_date(  ###for  delta_table only
        output_name,
        date_col=date_col,
        custom_config=custom_config,
        key_vault_dict=key_vault_dict,  ###for  delta_table only
        platform=platform,  ### for blob only
    )
    warn_txt = False
    if firstDate is not None:
        if isinstance(firstDate, str):
            print(firstDate)
            firstDate = dt.datetime.strptime(firstDate, "%Y-%m-%d").date()
        elif isinstance(firstDate, pd._libs.tslibs.timestamps.Timestamp):
            firstDate = firstDate.date()
        warn_txt = True
    else:
        firstDate = None if last_save_date is None else last_save_date + dt.timedelta(days=1)

    ###polish the warning, it is meaningless when last_save_date exists
    if (warn_txt) & (last_save_date is not None):
        print(
            f"last date is {last_save_date}; however, the function starts from given first date: {firstDate}"
        )

    udates = utils.monthly_first_dates(
        year_range=year_range,
        firstDate=firstDate,
        lastDate=lastDate,
        ###'2020-01-01'
    )

    if len(udates) == 0:
        print("Database|file is updated")
    else:
        print("date list updated to :\n", udates)

    for ii in range(len(udates) - 1):
        try:
            ##TODO: what if, there are more than on output, in that case output_name is only for the first output,
            start_date, end_date = utils.extract_start_end(udates, ii)
            ouputs_list = df_generator_func(
                start_date, end_date, output_name, **df_generator_func_args
            )

            save_outputs(  ###for  delta_table only
                ouputs_list
            )

        except Exception as e:
            print(f"***Creating Database(s) for {start_date} failed: \n\t\t {str(e)}")
            print(
                "**********************************************************************************************"
            )
            ##sys.exit()


# -------------------------------------------------------------------------
