"""List utilities: regex search, flatten, deduplicate, dropouts."""

import re

import numpy as np


def regex_filter_list(regLst, LstAll):
    """Return items of ``LstAll`` that match any regex in ``regLst``.

    Parameters
    ----------
    regLst : str or list of str
        One regex pattern or a list of patterns. Each is compiled and
        applied via ``re.search`` (partial match).
    LstAll : list of str
        Strings to filter.

    Returns
    -------
    out : list of str
        Items from ``LstAll`` matched by at least one pattern (order
        follows ``regLst`` then original order within each pattern).
    ind : numpy.ndarray of bool
        Boolean mask of length ``len(LstAll)`` marking each matched item.

    Examples
    --------
    >>> regex_filter_list(['.vol_flag$', 'fefefre', '_date'],
    ...                   ['bi_alt_account_id', 'snapshot_date',
    ...                    'snapshot_year', 'tv_vol_flag', 'phone_vol_flag'])
    (['tv_vol_flag', 'phone_vol_flag', 'snapshot_date'],
     array([False,  True, False,  True,  True]))
    """
    out = []
    if not isinstance(regLst, list):
        regLst = [regLst]
    for i in regLst:
        tmp = list(filter(re.compile(i).search, LstAll))
        out = out + tmp
    ind = np.isin(LstAll, out)
    return out, ind


def flatten_list(ulist):
    """Recursively flatten a list of arbitrarily nested lists.

    Parameters
    ----------
    ulist : list
        Nested list; non-list items are treated as leaves.

    Returns
    -------
    list
        Flat list preserving left-to-right order of leaves.
    """
    results = []
    for rec in ulist:
        if isinstance(rec, list):
            results.extend(rec)
            results = flatten_list(results)
        else:
            results.append(rec)
    return results


def unique_list(seq):
    """Return unique items of ``seq`` preserving first-seen order.

    Parameters
    ----------
    seq : iterable
        Input sequence, possibly with duplicates.

    Returns
    -------
    list
        Deduplicated items in original order.
    """
    seen = set()
    seen_add = seen.add
    out = [x for x in seq if not (x in seen or seen_add(x))]
    return out


def remove_extra_none(nested_lst):
    """Deduplicate a list and drop the literal string ``'None'`` if other values exist.

    Parameters
    ----------
    nested_lst : list
        Input list, possibly with duplicates and ``'None'`` entries.

    Returns
    -------
    list
        Order-preserving unique items, with ``'None'`` removed unless it
        is the only value.
    """
    items = list(dict.fromkeys(nested_lst))
    if ("None" in items) & (len(items) > 1):
        items.remove("None")
    # print(items)
    return items
