"""SQL parsing helpers: split a multi-statement SQL blob, read sql files."""

from .paths import check_path


def strip_sql_comments(text, *, drop_blank_lines=True):
    """Return ``text`` with SQL comments removed.

    Strips both ``--`` line comments (through end of line) and
    ``/* ... */`` block comments. String literals in single or double
    quotes are respected, so a ``--`` or ``/*`` inside a quoted string is
    left untouched. Backslash escapes inside strings are honored the same
    way :func:`_split_sql_expressions` handles them.

    The function is intended for producing a clean copy of a query for
    display or logging — it does not modify the query semantics, so the
    original ``text`` should still be sent to the database.

    Parameters
    ----------
    text : str
        Raw SQL text, possibly with comments.
    drop_blank_lines : bool, default True
        If True, drop lines that are empty after comments are removed.
        If False, keep them (useful when preserving the original line
        numbers matters).

    Returns
    -------
    str
        SQL text with comments removed.

    Examples
    --------
    >>> sql = '''
    ... -- header comment
    ... select 1 as a  -- trailing comment
    ... /* block
    ...    comment */
    ... from dual;
    ... '''
    >>> print(strip_sql_comments(sql))
    select 1 as a
    from dual;
    """
    out = []
    state = None  # None | '-' | '--' | '/' | '/*' | '/**' | '"' | "'" | '"\\' | "'\\"
    for c in text:
        if state is None:
            if c in "\"'":
                out.append(c)
                state = c
            elif c == "-":
                # Might be the first '-' of '--'. Buffer decision until next char.
                state = "-"
            elif c == "/":
                # Might be the first '/' of '/*'. Buffer decision until next char.
                state = "/"
            else:
                out.append(c)
        elif state == "-":
            if c == "-":
                # Confirmed '--' line comment; do not emit either dash.
                state = "--"
            else:
                # Not a comment after all; emit the buffered dash then handle c.
                out.append("-")
                state = None
                # Re-dispatch this char through the default branch.
                if c in "\"'":
                    out.append(c)
                    state = c
                elif c == "-":
                    state = "-"
                elif c == "/":
                    state = "/"
                else:
                    out.append(c)
        elif state == "--":
            # Inside line comment: drop chars until newline (emit the newline).
            if c == "\n":
                out.append(c)
                state = None
        elif state == "/":
            if c == "*":
                # Confirmed '/*' block comment; do not emit the slash or star.
                state = "/*"
            else:
                out.append("/")
                state = None
                if c in "\"'":
                    out.append(c)
                    state = c
                elif c == "-":
                    state = "-"
                elif c == "/":
                    state = "/"
                else:
                    out.append(c)
        elif state == "/*":
            if c == "*":
                state = "/**"
        elif state == "/**":
            if c == "/":
                state = None
            elif c != "*":
                state = "/*"
        elif state[0] in "\"'":
            # Inside a string literal — always emit, watch for terminator.
            out.append(c)
            if state.endswith("\\"):
                # Previous char was a backslash; consume this char literally.
                state = state[0]
            elif c == "\\":
                state = state[0] + "\\"
            elif c == state[0]:
                state = None
        else:
            raise ValueError(f"Illegal state {state!r} while stripping SQL comments")

    # Flush any trailing buffered single char (unmatched leading '-' or '/').
    if state == "-":
        out.append("-")
    elif state == "/":
        out.append("/")

    result = "".join(out)
    if drop_blank_lines:
        result = "\n".join(line for line in result.splitlines() if line.strip())
    return result


def _split_sql_expressions(text):
    """Split a multi-statement SQL blob on ``;`` while respecting quotes and comments.

    Handles single/double-quoted strings, ``--`` line comments, and
    ``/* ... */`` block comments so semicolons inside them are ignored.

    Parameters
    ----------
    text : str
        Raw SQL text.

    Returns
    -------
    list of str
        Individual SQL statements, stripped of the terminating ``;`` and
        surrounding whitespace; empty statements are dropped.
    """
    # from riskmodelPipeline.py
    results = []
    current = ""
    state = None
    for c in text:
        if state is None:  # default state, outside of special entity
            current += c
            if c in "\"'":
                # quoted string
                state = c
            elif c == "-":
                # probably "--" comment
                state = "-"
            elif c == "/":
                # probably '/*' comment
                state = "/"
            elif c == ";":
                # remove it from the statement
                current = current[:-1].strip()
                # and save current stmt unless empty
                if current:
                    results.append(current)
                current = ""
        elif state == "-":
            if c != "-":
                # not a comment
                state = None
                current += c
                continue
            # remove first minus
            current = current[:-1]
            # comment until end of line
            state = "--"
        elif state == "--":
            if c == "\n":
                # end of comment
                # and we do include this newline
                current += c
                state = None
            # else just ignore
        elif state == "/":
            if c != "*":
                state = None
                current += c
                continue
            # remove starting slash
            current = current[:-1]
            # multiline comment
            state = "/*"
        elif state == "/*":
            if c == "*":
                # probably end of comment
                state = "/**"
        elif state == "/**":
            if c == "/":
                state = None
            else:
                # not an end
                state = "/*"
        elif state[0] in "\"'":
            current += c
            if state.endswith("\\"):
                # prev was backslash, don't check for ender
                # just revert to regular state
                state = state[0]
                continue
            elif c == "\\":
                # don't check next char
                state += "\\"
                continue
            elif c == state[0]:
                # end of quoted string
                state = None
        else:
            raise Exception("Illegal state %s" % state)

    if current:
        current = current.rstrip(";").strip()
        if current:
            results.append(current)
    return results


def parse_sql_file(file_name):
    """Read a ``.sql`` file and return its statements as a list.

    Wraps :func:`_split_sql_expressions` on the file contents.

    Parameters
    ----------
    file_name : str
        Path to a SQL file.

    Returns
    -------
    list of str
        One statement per element (semicolons and surrounding
        whitespace stripped).

    Raises
    ------
    argparse.ArgumentTypeError
        Propagated from :func:`check_path` if the file does not exist.
    """
    # from riskmodelPipeline.py
    check_path(file_name)
    sql_statements = []
    with open(file_name) as f:
        for sql_statement in _split_sql_expressions(f.read()):
            sql_statements.append(sql_statement)
    return sql_statements
