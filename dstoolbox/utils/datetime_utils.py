"""Date/time utilities: timestamp validation, day counting, monthly date list generation."""

import time
import itertools
import datetime as dt
import pandas as pd


def check_timestamps(start, end, format_required='%Y-%m-%d'):
    """Return True if both ``start`` and ``end`` parse as ``format_required``.

    Parameters
    ----------
    start, end : str
        Candidate timestamp strings.
    format_required : str, optional
        ``strptime`` format string; defaults to ``'%Y-%m-%d'``.

    Returns
    -------
    bool or None
        ``True`` when both inputs parse successfully; ``None`` (with the
        parse error printed) on failure.
    """
    try:
        check_start = type(time.strptime(start, format_required))
        check_end = type(time.strptime(end, format_required))
        if (check_start.__name__=='struct_time') & (check_end.__name__=='struct_time'):
            return True
    except ValueError as e:
        print(e)


def pass_days(start_date, end_date):
    """Count days within ``[start_date, end_date]`` per calendar quarter.

    Parameters
    ----------
    start_date, end_date : str or datetime-like
        Inclusive bounds; anything ``pd.date_range`` accepts.

    Returns
    -------
    pandas.Series
        Index is ``"Q1"..."Q4"`` and values are the number of days from
        the input range falling in that quarter (across the years
        spanned by the range).
    """
    ##TODO: add comment

    # if (start_date is None)|(end_date is None)| (pd.isnull(start_date))| (pd.isnull(end_date)):
    #   return None
    month_year_index = (
        pd.date_range(start=start_date, end=end_date, freq="D").to_period("Q").unique()
    )
    # print(start_date, end_date, month_year_index)

    pass_days_dict = {}
    for month_year in month_year_index:
        days_in_month = (
            min(end_date, month_year.end_time)
            - max(start_date, (month_year.start_time))
        ).days + 1
        pass_days_dict[month_year] = days_in_month

    result_series = pd.Series(pass_days_dict)
    qs = "Q" + result_series.index.quarter.astype(str)
    result_series = result_series.groupby(qs).sum()
    # print(result_series)
    return result_series.fillna(0)


def seconds_to_dhms(time):
    """Split a duration in seconds into ``(days, hours, minutes, seconds)``.

    Parameters
    ----------
    time : int or float
        Duration in seconds.

    Returns
    -------
    tuple of int/float
        ``(day, hour, minutes, seconds)``. The final seconds component
        keeps the fractional part of the input.
    """
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    return (day, hour, minutes, seconds)


def monthly_first_dates(year_range=None,
              firstDate=None,
              lastDate=None,
              month_step=1):
  """Generate a list of ``YYYY-MM-01`` dates within a year range, clipped to ``[firstDate, lastDate]``.

  Parameters
  ----------
  year_range : list of int or None, optional
      Two-element ``[start_year, stop_year]`` (stop is exclusive).
      Defaults to ``[2018, 2099]``.
  firstDate : str or datetime-like or None, optional
      Lower bound (inclusive). Defaults to the earliest date derived
      from ``year_range``.
  lastDate : str or datetime-like or None, optional
      Upper bound (inclusive). Defaults to today.
  month_step : int, optional
      Month stride between generated dates. Default ``1`` (every month).

  Returns
  -------
  list of str
      ``YYYY-MM-DD`` strings, one per selected month plus ``lastDate``
      if it isn't already the last first-of-month.
  """
  if year_range is None:
    year_range = [2018, 2099]
  if lastDate is None:
    lastDate = dt.datetime.now().date()

  # print(firstDate)
  # print(lastDate)
  yrs=[str(i) for i in range(year_range[0], year_range[1])]
  months=[str(i).zfill(2) for i in range(1,13, month_step)]
  udates=['-'.join(udate) for udate in itertools.product(yrs,months,['01'])]

  if isinstance(firstDate, str):
    print(firstDate)
    firstDate   = dt.datetime.strptime(firstDate, "%Y-%m-%d").date()
  elif isinstance(firstDate,pd._libs.tslibs.timestamps.Timestamp):
    firstDate   =firstDate.date()
  if isinstance(lastDate, str):
    lastDate     = dt.datetime.strptime(lastDate, "%Y-%m-%d").date()
  elif isinstance(lastDate,pd._libs.tslibs.timestamps.Timestamp):
    lastDate   =lastDate.date()

  if firstDate is None:
    firstDate=dt.datetime.strptime(udates[0], '%Y-%m-%d').date()
  if lastDate is None:
    lastDate=dt.datetime.strptime(udates[-1], '%Y-%m-%d').date()

  if udates[-1]!=lastDate:
    udates.append(lastDate.strftime("%Y-%m-%d"))
  if udates[0]!=firstDate:
    udates[0]=firstDate.strftime("%Y-%m-%d")
    # udates.insert(0,firstDate.strftime("%Y-%m-%d"))

  udates=[ii for ii in udates if (dt.datetime.strptime(ii, '%Y-%m-%d').date()>=firstDate)&\
                                 (dt.datetime.strptime(ii, '%Y-%m-%d').date()<=lastDate)]
  # print(udates)
  return udates
