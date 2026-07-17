"""SQL parsing helpers: split a multi-statement SQL blob, read sql files."""

from .paths import check_path


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
    current = ''
    state = None
    for c in text:
        if state is None:  # default state, outside of special entity
            current += c
            if c in '"\'':
                # quoted string
                state = c
            elif c == '-':
                # probably "--" comment
                state = '-'
            elif c == '/':
                # probably '/*' comment
                state = '/'
            elif c == ';':
                # remove it from the statement
                current = current[:-1].strip()
                # and save current stmt unless empty
                if current:
                    results.append(current)
                current = ''
        elif state == '-':
            if c != '-':
                # not a comment
                state = None
                current += c
                continue
            # remove first minus
            current = current[:-1]
            # comment until end of line
            state = '--'
        elif state == '--':
            if c == '\n':
                # end of comment
                # and we do include this newline
                current += c
                state = None
            # else just ignore
        elif state == '/':
            if c != '*':
                state = None
                current += c
                continue
            # remove starting slash
            current = current[:-1]
            # multiline comment
            state = '/*'
        elif state == '/*':
            if c == '*':
                # probably end of comment
                state = '/**'
        elif state == '/**':
            if c == '/':
                state = None
            else:
                # not an end
                state = '/*'
        elif state[0] in '"\'':
            current += c
            if state.endswith('\\'):
                # prev was backslash, don't check for ender
                # just revert to regular state
                state = state[0]
                continue
            elif c == '\\':
                # don't check next char
                state += '\\'
                continue
            elif c == state[0]:
                # end of quoted string
                state = None
        else:
            raise Exception('Illegal state %s' % state)

    if current:
        current = current.rstrip(';').strip()
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
    with open(file_name, 'r') as f:
        for sql_statement in _split_sql_expressions(f.read()):
            sql_statements.append(sql_statement)
    return(sql_statements)
