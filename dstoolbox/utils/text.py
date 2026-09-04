"""Text utilities: normalization, sanitization, fuzzy matching, list comparison, run-length encoding."""

import inspect
import re
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle


def normalize_text(
    text,
    remove_spaces=True,
    lowercase=True,
    special_chars=r"[^a-zA-Z0-9\s]",
    replace_with="",
    max_length=None,
    fallback_text="unnamed",
):
    """Normalize a string: lowercase, strip special characters, collapse spaces, cap length.

    Parameters
    ----------
    text : Any
        Value to normalize; non-strings are coerced via ``str()``.
    remove_spaces : bool, optional
        If True (default), replace whitespace runs with ``replace_with``.
    lowercase : bool, optional
        If True (default), lowercase the text.
    special_chars : str or None, optional
        Regex of characters to strip; default keeps only letters,
        digits, and whitespace. Pass a falsy value to skip.
    replace_with : str, optional
        Replacement for matched ``special_chars`` and whitespace runs.
        Default ``''`` (removal).
    max_length : int or None, optional
        Truncate the result to this length. Default ``None``.
    fallback_text : str, optional
        Returned when the result is empty or whitespace-only. Default
        ``'unnamed'``.

    Returns
    -------
    str
        Normalized text (or ``fallback_text``).
    """
    if not isinstance(text, str):
        text = str(text)

    if lowercase:
        text = text.lower()

    if special_chars:
        text = re.sub(special_chars, replace_with, text)

    if remove_spaces:
        text = re.sub(r"\s+", replace_with, text)

    text = text.strip()

    # Trim text if max_length is specified and text exceeds it
    if max_length is not None and len(text) > max_length:
        text = text[:max_length]
        # Strip again after truncation in case we cut mid-word
        text = text.strip()

    # Handle empty or whitespace-only results
    if not text or text.isspace():
        text = fallback_text

    return text


def clean_column_names(column_list, replacements=None, lowercase=False):
    """Sanitize a list of strings into DataFrame-column-safe identifiers.

    Each name is normalized via :func:`normalize_text` (keeping only
    ``[A-Za-z0-9_]``), then duplicate underscores are collapsed and
    leading/trailing underscores stripped. Names starting with a digit
    are prefixed with ``'col_'``.

    Parameters
    ----------
    column_list : list
        Candidate column names (any type; coerced to ``str``).
    replacements : dict or None, optional
        Explicit ``{old: new}`` substitutions applied before
        normalization. Default ``None`` (no substitutions).
    lowercase : bool, optional
        Lowercase before normalization. Default ``False``.

    Returns
    -------
    list of str
        Cleaned column names of the same length as ``column_list``.
    """
    if replacements is None:
        replacements = {}
    cleaned_columns = []

    for col in column_list:
        # Convert to string if not already
        col = str(col)

        # Apply custom replacements first
        for old, new in replacements.items():
            col = col.replace(old, new)

        # Use normalize_text for most of the work
        col = normalize_text(
            col,
            remove_spaces=True,
            lowercase=lowercase,
            special_chars=r"[^a-zA-Z0-9_]",
            replace_with="_",
            fallback_text="unnamed_column",
        )

        # Collapse multiple underscores and strip leading/trailing ones
        col = re.sub(r"_+", "_", col).strip("_")

        # Ensure column doesn't start with a number
        if col and col[0].isdigit():
            col = "col_" + col

        # Final fallback
        if not col:
            col = "unnamed_column"

        cleaned_columns.append(col)

    return cleaned_columns


def find_fuzzy_matches(listA, listB, threshold=60):
    """Greedy fuzzy-match items of ``listA`` to items of ``listB`` after normalization.

    Uses :func:`normalize_text` for canonicalization and
    ``difflib.SequenceMatcher`` for similarity. Each ``listB`` item can
    be matched at most once (greedy: first best match wins).

    Parameters
    ----------
    listA, listB : list
        Candidate strings to match across.
    threshold : float, optional
        Minimum similarity percentage (0–100) required to accept a
        pair. Default ``60``.

    Returns
    -------
    dict
        ``{item_a: {'match', 'similarity', 'normalized_a',
        'normalized_b'}}`` for every ``listA`` item that found a
        qualifying partner.
    """
    matches = {}
    used_b_indices = set()

    for i, item_a in enumerate(listA):
        normalized_a = normalize_text(item_a)
        best_match = None
        best_similarity = 0
        best_index = -1

        for j, item_b in enumerate(listB):
            if j in used_b_indices:
                continue

            normalized_b = normalize_text(item_b)
            similarity = SequenceMatcher(None, normalized_a, normalized_b).ratio() * 100

            if similarity >= threshold and similarity > best_similarity:
                best_match = item_b
                best_similarity = similarity
                best_index = j

        if best_match:
            matches[item_a] = {
                "match": best_match,
                "similarity": best_similarity,
                "normalized_a": normalized_a,
                "normalized_b": normalize_text(best_match),
            }
            used_b_indices.add(best_index)

    return matches


def create_venn_diagram(
    listA,
    listB,
    similarity_threshold=60,
    listA_name="List A",
    listB_name="List B",
    utitle="Venn Diagram - List Comparison with Fuzzy Matching",
    save_path=None,
):
    """Two-set Venn diagram of ``listA`` vs ``listB`` after fuzzy matching.

    Overlap size is the number of ``listA``-``listB`` pairs whose
    fuzzy similarity meets ``similarity_threshold``; complements are
    the leftover items in each list.

    Parameters
    ----------
    listA, listB : list
        Two datasets to compare.
    similarity_threshold : float, optional
        Percentage threshold for :func:`find_fuzzy_matches`. Default
        ``60``.
    listA_name, listB_name : str, optional
        Display names for the two circles.
    utitle : str, optional
        Figure title.
    save_path : str or None, optional
        If given, the figure is saved to this path at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered Venn diagram figure.
    """
    # Find fuzzy matches using the existing function
    fuzzy_matches = find_fuzzy_matches(listA, listB, similarity_threshold)
    # Calculate counts
    matched_a = set(fuzzy_matches.keys())
    matched_b = set(match_info["match"] for match_info in fuzzy_matches.values())
    unmatched_a = set(listA) - matched_a
    unmatched_b = set(listB) - matched_b

    only_a_count = len(unmatched_a)
    only_b_count = len(unmatched_b)
    both_count = len(fuzzy_matches)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Define circle parameters
    circle_radius = 1.5
    circle_a_center = (-0.5, 0)
    circle_b_center = (0.5, 0)

    # Create circles
    circle_a = Circle(circle_a_center, circle_radius, alpha=0.3, color="blue", label=listA_name)
    circle_b = Circle(circle_b_center, circle_radius, alpha=0.3, color="red", label=listB_name)

    ax.add_patch(circle_a)
    ax.add_patch(circle_b)

    # Add text labels with counts
    # Only A
    ax.text(
        circle_a_center[0] - 0.8,
        circle_a_center[1],
        f"{only_a_count}",
        fontsize=16,
        ha="center",
        va="center",
        weight="bold",
    )

    # Only B
    ax.text(
        circle_b_center[0] + 0.8,
        circle_b_center[1],
        f"{only_b_count}",
        fontsize=16,
        ha="center",
        va="center",
        weight="bold",
    )

    # Both (intersection)
    ax.text(0, 0, f"{both_count}", fontsize=16, ha="center", va="center", weight="bold")

    # Add circle labels
    ax.text(
        circle_a_center[0],
        circle_a_center[1] + 2,
        listA_name,
        fontsize=14,
        ha="center",
        va="center",
        weight="bold",
        color="blue",
    )
    ax.text(
        circle_b_center[0],
        circle_b_center[1] + 2,
        listB_name,
        fontsize=14,
        ha="center",
        va="center",
        weight="bold",
        color="red",
    )

    # Add detailed breakdown text
    breakdown_text = "Breakdown:\n"
    breakdown_text += f"• Only in {listA_name}: {only_a_count} items\n"
    breakdown_text += f"• Only in {listB_name}: {only_b_count} items\n"
    breakdown_text += f"• Similar items (≥{similarity_threshold}%): {both_count} pairs\n"
    breakdown_text += f"• Total unique items: {only_a_count + only_b_count + both_count}"

    ax.text(
        -3,
        -3,
        breakdown_text,
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    # Set axis properties
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add title
    ax.set_title(utitle, fontsize=16, weight="bold", pad=20)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def compare_lists(
    listA,
    listB,
    similarity_threshold=60,
    create_venn=False,
    listA_name="List A",
    listB_name="List B",
    utitle="Venn Diagram - List Comparison with Fuzzy Matching",
    save_venn_path=None,
):
    """Fuzzy-compare two lists and return a per-element attribution table + summary.

    Every element of ``listA`` or ``listB`` becomes one row tagged as
    ``listA_name``, ``listB_name``, or ``'Both'`` (fuzzy match above
    ``similarity_threshold``). Optionally also renders a Venn diagram
    via :func:`create_venn_diagram`.

    Parameters
    ----------
    listA, listB : list
        Two datasets to compare.
    similarity_threshold : float, optional
        Percentage threshold used by :func:`find_fuzzy_matches`.
        Default ``60``.
    create_venn : bool, optional
        If True, also build a Venn diagram. Default ``False``.
    listA_name, listB_name : str, optional
        Display names for the two lists (used in the ``Group`` column
        and, if requested, in the Venn diagram).
    utitle : str, optional
        Title used only when ``create_venn`` is True.
    save_venn_path : str or None, optional
        Save path forwarded to :func:`create_venn_diagram`.

    Returns
    -------
    tuple
        ``(result_df, summary_text, venn_fig)``: unified table, a
        multi-line report string, and either the Venn figure or
        ``None``.
    """
    # Find fuzzy matches
    fuzzy_matches = find_fuzzy_matches(listA, listB, similarity_threshold)

    # Create sets for tracking matched items
    matched_a = set(fuzzy_matches.keys())
    matched_b = set(match_info["match"] for match_info in fuzzy_matches.values())

    # Find unmatched items
    unmatched_a = set(listA) - matched_a
    unmatched_b = set(listB) - matched_b

    # Create list of dictionaries for DataFrame
    data = []

    # Add elements only in listA (unmatched)
    for element in sorted(unmatched_a):
        data.append({"Element": element, "Group": listA_name, "Match": "", "Similarity": ""})

    # Add elements only in listB (unmatched)
    for element in sorted(unmatched_b):
        data.append({"Element": element, "Group": listB_name, "Match": "", "Similarity": ""})

    # Add fuzzy matched elements
    for item_a, match_info in sorted(fuzzy_matches.items()):
        data.append(
            {
                "Element": item_a,
                "Group": "Both",
                "Match": match_info["match"],
                "Similarity": f"{match_info['similarity']:.1f}%",
            }
        )

    # Create DataFrame
    result_df = pd.DataFrame(data)

    # Sort by Group first, then by Element
    if not result_df.empty:
        result_df = result_df.sort_values(["Group", "Element"], ascending=[True, True])
        result_df = result_df.reset_index(drop=True)
        result_df.insert(0, "Index", range(1, len(result_df) + 1))

    # Create formatted summary
    summary_text = "Unique Elements Table with Fuzzy Matching:\n"
    summary_text += "=" * 60 + "\n"
    summary_text += result_df.to_string(index=False) + "\n"
    summary_text += "\nGroup Summary:\n"
    summary_text += "=" * 20 + "\n"

    if not result_df.empty:
        group_counts = result_df["Group"].value_counts()
        for group, count in group_counts.items():
            summary_text += f"{group}: {count} elements\n"

    summary_text += "\nFuzzy Matching Details:\n"
    summary_text += "=" * 25 + "\n"
    summary_text += f"Similarity threshold: {similarity_threshold}%\n"
    summary_text += f"Fuzzy matches found: {len(fuzzy_matches)}\n"

    if fuzzy_matches:
        summary_text += "\nDetailed Matches:\n"
        for item_a, match_info in sorted(fuzzy_matches.items()):
            summary_text += (
                f"  '{item_a}' ↔ '{match_info['match']}' ({match_info['similarity']:.1f}%)\n"
            )
            summary_text += (
                f"    Normalized: '{match_info['normalized_a']}' ↔ '{match_info['normalized_b']}'\n"
            )

    summary_text += "\nOriginal Data:\n"
    summary_text += f"listA: {listA}\n"
    summary_text += f"listB: {listB}"

    # Create Venn diagram if requested
    venn_fig = None
    if create_venn:
        venn_fig = create_venn_diagram(
            listA, listB, similarity_threshold, listA_name, listB_name, utitle, save_venn_path
        )

    return result_df, summary_text, venn_fig


def retrieve_name(var):
    """Best-effort lookup of the caller-local variable name bound to ``var``.

    Walks the caller's locals looking for an ``is``-identical binding.
    Returns the first match; behaviour is undefined if the same object
    is bound to multiple names.

    Parameters
    ----------
    var : object
        Value whose local name should be recovered.

    Returns
    -------
    str
        Variable name in the caller's frame.

    References
    ----------
    https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def rle_encode(data):
    """Run-length encode an iterable into ``"<char>(<count>);"`` segments.

    Parameters
    ----------
    data : iterable
        Sequence of comparable items (typically a string).

    Returns
    -------
    str
        Encoding such as ``"a(3);b(2);c(1);"`` for ``"aaabbc"``;
        empty string when ``data`` is empty.

    References
    ----------
    https://stackabuse.com/run-length-encoding/
    """
    # Ref:https://stackabuse.com/run-length-encoding/
    encoding = ""
    prev_char = ""
    count = 1

    if not data:
        return ""

    for char in data:
        if char != prev_char:
            if prev_char:
                encoding += prev_char + "(" + str(count) + ");"
            count = 1
            prev_char = char
        else:
            count += 1
    encoding += prev_char + "(" + str(count) + ");"
    return encoding
