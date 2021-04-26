"""Tests for logger."""
import logging
import os
from pathlib import Path

import pytest

from deepreg import log


@pytest.fixture()
def log_file_path(tmp_path_factory) -> Path:
    """
    Fixture for temporary log file.

    :param tmp_path_factory: default fixture.
    :return: a temporary file path
    """
    return tmp_path_factory.mktemp("test_log") / "log.txt"


@pytest.mark.parametrize("log_level", [0, 1, 2, 3, 4, 5])
def test_get(log_level: int, log_file_path: Path):
    """
    Test loggers by count number of logs.

    :param log_level: 0 to 5.
    :param log_file_path: path of a temporary log file.
    """
    os.environ["DEEPREG_LOG_LEVEL"] = str(log_level)
    logger = log.get(__name__)

    # save logs into a file
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger.debug("DEBUG")
    logger.info("INFO")
    logger.warning("WARNING")
    logger.error("ERROR")
    logger.critical("CRITICAL")

    # count lines in the file
    with open(log_file_path, "r") as f:
        num_logs = len(f.readlines())

    assert num_logs == 6 - max(1, log_level)
