"""PI Web API ingestion: WebID lookup, interpolated/raw/per-second time-series fetch."""

import datetime as dt

import pandas as pd

from .. import default_values as par
from . import data_sources

# PI Web API server + data-server path. Configure these via ``pi2pd_*`` kwargs
# for your deployment; the defaults below are placeholders only.
DEFAULT_PI_WEBAPI_BASE = "https://<pi-webapi-host>/piwebapi"
DEFAULT_PI_DATA_SERVER = "<PI-DATA-SERVER>"
DEFAULT_TZ = "UTC"

# Seconds to wait on any single PI Web API request before giving up.
HTTP_TIMEOUT_SECONDS = 60


def get_web_ids(
    access_token: str,
    tags,
    webapi_base: str = DEFAULT_PI_WEBAPI_BASE,
    data_server: str = DEFAULT_PI_DATA_SERVER,
):
    """Resolve a list of PI tags to their persistent WebIDs.

    Parameters
    ----------
    access_token : str
        OAuth bearer token (from ``data_sources.get(id).bearer_token``).
    tags : Iterable[str]
        PI tag names to resolve.
    webapi_base : str, optional
        Base URL of the PI Web API (e.g.
        ``https://pi.example.com/piwebapi``).
    data_server : str, optional
        PI data-server name used in the tag path.

    Returns
    -------
    dict
        ``{tag: WebID}``. Tags that could not be resolved map to ``None``.
    """
    import json

    import requests

    web_ids = {}
    headers = {
        "Authorization": "Bearer " + access_token,
        "Content-Type": "application/json",
    }
    for tag in tags:
        url = f"{webapi_base}/points?path=\\\\{data_server}\\{tag}"
        response = requests.request("GET", url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
        web_id = json.loads(response.text).get("WebId")
        web_ids[tag] = web_id
        if web_id is None:
            print(f"PI tag not found: {tag}")
    return web_ids


def _parse_date(d):
    if isinstance(d, str):
        return dt.datetime.strptime(d, "%Y-%m-%d")
    return d


def pi2pd_interpolate(
    tags,
    start_date=par.start_date,
    end_date=par.end_date,
    interval: str = "1h",
    target_id: str = "webapi",
    runtime: str = "databricks",
    webapi_base: str = DEFAULT_PI_WEBAPI_BASE,
    data_server: str = DEFAULT_PI_DATA_SERVER,
    timezone: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """Fetch PI tag data resampled at ``interval`` between two dates.

    Parameters
    ----------
    tags : str or list of str
        PI tag name(s) to fetch. A single string is wrapped into a list.
    start_date, end_date : str or datetime
        Time window. Strings must be ``YYYY-MM-DD``.
    interval : str, default ``'1h'``
        Resampling interval passed to the PI Web API (e.g. ``'1s'``,
        ``'1m'``, ``'1h'``, ``'1d'``).
    target_id : str, default ``'webapi'``
        Key into ``data_sources`` used to obtain the bearer token.
    runtime : str, default ``'databricks'``
        Passed to ``data_sources.get`` to select the credential path.
    webapi_base : str, optional
        PI Web API base URL.
    data_server : str, optional
        PI data-server name.
    timezone : str, optional
        Time zone the returned ``Date`` column is converted to.

    Returns
    -------
    pandas.DataFrame
        Columns: ``Date`` plus one column per resolved tag.
    """
    import json
    import urllib

    import requests

    start_date = _parse_date(start_date)
    end_date = _parse_date(end_date)
    if not isinstance(tags, list):
        tags = [tags]

    access_token = data_sources.get(target_id, runtime=runtime).bearer_token
    web_ids = get_web_ids(access_token, tags, webapi_base=webapi_base, data_server=data_server)
    tag_data = {}
    headers = {
        "Authorization": "Bearer " + access_token,
        "Content-Type": "application/json",
    }
    for i, tag in enumerate(tags):
        web_id = web_ids[tag]
        if web_id is None:
            continue
        print(
            "tag=",
            tag,
            ",webID=",
            f"{web_id[:5]}...{web_id[20:25]}...{web_id[-5:]}",
        )
        params = {"startTime": start_date, "endTime": end_date, "interval": interval}
        url = f"{webapi_base}/streams/{web_id}/interpolated?{urllib.parse.urlencode(params)}"
        response = requests.request("GET", url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
        body = json.loads(response.text)
        if i == 0:
            tag_data["Date"] = [j["Timestamp"] for j in body["Items"]]
        first_value = body["Items"][0]["Value"]
        if isinstance(first_value, dict):
            tag_data[tag] = [
                j["Value"]["Name"] if isinstance(j["Value"], dict) else j["Value"]
                for j in body["Items"]
            ]
        else:
            tag_data[tag] = [j["Value"] for j in body["Items"]]

    df = pd.DataFrame(tag_data, columns=tag_data.keys())
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_convert(timezone)
    return df


def pi2pd_raw_data(
    tags,
    start_date=par.start_date,
    end_date=par.end_date,
    target_id: str = "webapi",
    runtime: str = "databricks",
    webapi_base: str = DEFAULT_PI_WEBAPI_BASE,
    data_server: str = DEFAULT_PI_DATA_SERVER,
    timezone: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """Fetch PI tag data at the original recording frequency (unsampled).

    Uses the ``/recorded`` endpoint; skips tags whose HTTP response is not
    200. "Bad" values are replaced with ``None``.

    Parameters
    ----------
    tags : str or list of str
        PI tag name(s) to fetch.
    start_date, end_date : str or datetime
        Time window. Strings must be ``YYYY-MM-DD``.
    target_id, runtime, webapi_base, data_server, timezone
        See :func:`pi2pd_interpolate`.

    Returns
    -------
    pandas.DataFrame
        Long-form frame with ``Timestamp`` plus one column per tag (rows
        are appended as raw samples arrive; expect NaNs across tags).
    """
    import json
    import urllib

    import requests

    start_date = _parse_date(start_date)
    end_date = _parse_date(end_date)
    if not isinstance(tags, list):
        tags = [tags]

    access_token = data_sources.get(target_id, runtime=runtime).bearer_token
    web_ids = get_web_ids(access_token, tags, webapi_base=webapi_base, data_server=data_server)
    headers = {
        "Authorization": "Bearer " + access_token,
        "Content-Type": "application/json",
    }
    entries = []
    for tag in tags:
        web_id = web_ids[tag]
        if web_id is None:
            continue
        print(
            "tag=",
            tag,
            ",webID=",
            f"{web_id[:5]}...{web_id[20:25]}...{web_id[-5:]}",
        )
        params = {"startTime": start_date, "endTime": end_date}
        url = f"{webapi_base}/streams/{web_id}/recorded?{urllib.parse.urlencode(params)}"
        response = requests.request("GET", url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
        if response.status_code != 200:
            continue
        body = json.loads(response.text)
        for item in body.get("Items", []):
            value = item["Value"]
            if isinstance(value, dict):
                value = None if value.get("Name") == "Bad" else value.get("Value")
            entries.append({"Timestamp": item["Timestamp"], tag: value})

    df = pd.DataFrame(entries)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_convert(timezone)
    return df


def pi2pd_seconds(
    tags,
    start_date=par.start_date,
    end_date=par.end_date,
    target_id: str = "webapi",
    runtime: str = "databricks",
    webapi_base: str = DEFAULT_PI_WEBAPI_BASE,
    data_server: str = DEFAULT_PI_DATA_SERVER,
    timezone: str = DEFAULT_TZ,
) -> pd.DataFrame:
    """Fetch PI tag data at one-second resolution, iterating day-by-day.

    Wraps :func:`pi2pd_interpolate` with ``interval='1s'`` and stitches
    day-sized windows to work around per-second-stream size limits
    (approximately 11 tags per request in typical deployments). Days that
    fail to fetch are skipped with a ``"Skipped"`` print.

    Parameters
    ----------
    tags : str or list of str
        PI tag name(s) to fetch.
    start_date, end_date : str or datetime
        Time window. Strings must be ``YYYY-MM-DD``.
    target_id, runtime, webapi_base, data_server, timezone
        See :func:`pi2pd_interpolate`.

    Returns
    -------
    pandas.DataFrame
        Concatenation of the per-day interpolated frames.
    """
    start_date = _parse_date(start_date)
    end_date = _parse_date(end_date)
    out = pd.DataFrame()
    cursor = start_date
    while cursor + dt.timedelta(days=1) <= end_date:
        window_end = cursor + dt.timedelta(days=1)
        print(f"getting data between {cursor} and {window_end}")
        try:
            chunk = pi2pd_interpolate(
                tags,
                start_date=cursor,
                end_date=window_end,
                interval="1s",
                target_id=target_id,
                runtime=runtime,
                webapi_base=webapi_base,
                data_server=data_server,
                timezone=timezone,
            )
            print("Done")
            out = pd.concat([out, chunk], ignore_index=True)
        except Exception:
            print("Skipped")
        cursor = window_end
    return out
