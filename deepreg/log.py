"""Module for logger."""
import logging
import os
import sys


def get(name: str) -> logging.Logger:
    """
    Configure the logger with formatter and handlers.

    The logger should be used as:

    .. code-block:: python

        from deepreg import log

        logger = log.get(__name__)

    The log level depends on the environment variable `DEEPREG_LOG_LEVEL`.

    - 0: NOTSET, will be set to DEBUG
    - 1: DEBUG
    - 2: INFO (default)
    - 3: WARNING
    - 4: ERROR
    - 5: CRITICAL

    https://docs.python.org/3/library/logging.html#levels

    :param name: module name.
    :return: configured logger.
    """
    logger = logging.getLogger(name=name)
    logger.propagate = False
    log_level = os.environ.get("DEEPREG_LOG_LEVEL", "2")
    log_level_int = max(int(log_level) * 10, 10)
    logger.setLevel(log_level_int)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(log_level_int)
    logger.addHandler(stdout_handler)
    return logger
