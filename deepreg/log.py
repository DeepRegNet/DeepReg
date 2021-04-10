"""Module for logger."""
import logging
import os
import sys


def get(name: str) -> logging.Logger:
    """
    Configure the logger with formatter and handlers.

    The log level depends on the environment variable `DEEPREG_LOG_LEVEL`.

    :param name: module name.
    :return: configured logger.
    """
    logger = logging.getLogger(name=name)
    logger.propagate = False
    logger.setLevel(os.environ.get("DEEPREG_LOG_LEVEL", "INFO"))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    return logger
