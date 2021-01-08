# coding=utf-8

"""
Tests for deepreg/download.py
pytest style
"""
import os
import shutil
from filecmp import dircmp

from git import Repo

from deepreg.download import main


def has_diff_files(dcmp):
    if len(dcmp.diff_files) > 0:
        return True
    for sub_dcmp in dcmp.subdirs.values():
        has_diff_files(sub_dcmp)
    return False


def test_download():
    # Covered by test_main
    pass


def test_main():
    """
    Integration test by checking the output dirs and files exist
    """

    temp_dir = "./deepreg_download_temp_dir"
    branch = Repo(".").head.object.hexsha

    main(args=["--output_dir", temp_dir, "--branch", branch])

    # Check downloading all req'd folders into temp, verify that they are the same as in main branch.
    config_dcmp = dircmp("./config", os.path.join(temp_dir, "config"))
    assert not has_diff_files(config_dcmp)

    data_dcmp = dircmp("./data", os.path.join(temp_dir, "data"))
    assert not has_diff_files(data_dcmp)

    demos_dcmp = dircmp("./demos", os.path.join(temp_dir, "demos"))
    assert not has_diff_files(demos_dcmp)

    shutil.rmtree(temp_dir)
