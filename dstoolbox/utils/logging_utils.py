"""Logging helpers: file+console logger creation, stdout/stderr redirect to log."""

import logging
import os
import sys
import warnings

_DEPRECATION_EMITTED = False


def custom_print(message, logger=None):
    """Log a message under the ``dstoolbox`` tree; also emit to stdout for back-compat.

    .. deprecated:: 0.4.0
        Prefer ``logging.getLogger(__name__)`` and call ``logger.info(...)`` /
        ``logger.warning(...)`` directly. This helper survives one release as a
        thin shim.

    Parameters
    ----------
    message : Any
        Object to log (anything with a ``__str__``).
    logger : logging.Logger or None, optional
        If supplied, ``logger.info(message)`` is also called. Otherwise the
        message is routed through the ``dstoolbox`` package logger.

    Returns
    -------
    None
    """
    global _DEPRECATION_EMITTED
    if not _DEPRECATION_EMITTED:
        warnings.warn(
            "custom_print is deprecated; use logging.getLogger(__name__).info/warning.",
            DeprecationWarning,
            stacklevel=2,
        )
        _DEPRECATION_EMITTED = True

    if logger is not None:
        logger.info(message)
    else:
        logging.getLogger("dstoolbox").info(message)


def make_logger(uFile, name, logLevel=logging.INFO):
    """Create a logger with both a file and a console handler.

    Idempotent: repeat calls with the same ``name`` do not add extra
    handlers.

    Parameters
    ----------
    uFile : str
        Path of the log file (opened with mode ``'w'`` — truncates on
        each call).
    name : str
        Logger name (as passed to ``logging.getLogger``).
    logLevel : int, optional
        Numeric level (``logging.INFO``, ``logging.DEBUG``, ...).
        Default ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured logger.

    Notes
    -----
    File records use ``'%(asctime)s | %(levelname)-3s | %(message)s'``;
    console records show the message only.

    Standard numeric levels: CRITICAL=50, ERROR=40, WARNING=30,
    INFO=20, DEBUG=10, NOTSET=0.
    """

    #   logging.basicConfig(filemode='w')

    # configure log formatter
    logFormatter1 = logging.Formatter(
        "%(asctime)s | %(levelname)-3s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logFormatter2 = logging.Formatter("%(message)s")
    #   logFormatter2 = logFormatter1

    # configure file handler
    fileHandler = logging.FileHandler(uFile, "w")
    fileHandler.setFormatter(logFormatter1)

    # configure stream handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter2)

    # get the logger instance
    logger = logging.getLogger(name)

    # set the logging level
    logger.setLevel(logLevel)

    if not len(logger.handlers):
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)
    return logger


class _loggerWriter:
    """Internal file-like adapter used by :func:`redirect_stdio_to_logger` to route ``stdout``/``stderr`` writes into a logger.

    Reference: https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    """

    def __init__(self, level):
        """Bind the logger callable used as the write target.

        Parameters
        ----------
        level : callable
            A logger method such as ``logger.info`` or ``logger.error``;
            invoked with each non-empty line written to this stream.
        """
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        """Forward a non-newline message to the configured log-level callable."""
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != "\n":
            self.level(message)

    def flush(self):
        """Stream-flush hook required by the file-like protocol; logs ``sys.stderr``."""
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)


def redirect_stdio_to_logger(stdoutLvl, stderrLvl):
    """Redirect ``sys.stdout`` and ``sys.stderr`` to log-level callables.

    Parameters
    ----------
    stdoutLvl : callable
        Log-level method (e.g. ``logger.info``) that receives each line
        written to ``sys.stdout``.
    stderrLvl : callable
        Log-level method (e.g. ``logger.error``) for lines written to
        ``sys.stderr``.

    Returns
    -------
    tuple
        ``(old_stdout, old_stderr)`` — the original streams, so the
        caller can restore them later.
    """

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _loggerWriter(stdoutLvl)
    sys.stderr = _loggerWriter(stderrLvl)
    return (old_stdout, old_stderr)


def setup_logger(log_file):
    """
    Set up logger to write to console and file
    Args:
        log_file (str): Path to the log file
    Returns:
        logging.Logger: Configured logger instance
    """

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:  # Only create directory if log_file includes a path
        os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger("intranet_downloader")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(message)s")

    # File handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file at {log_file}: {str(e)}")
        print("Continuing with console logging only...")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
