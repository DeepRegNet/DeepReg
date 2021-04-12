"""
Tests functions in config/parser.py
"""

import os

import pytest
import yaml
from testfixtures import TempDirectory

from deepreg.config.parser import (
    config_sanity_check,
    load_configs,
    save,
    update_nested_dict,
)


def test_update_nested_dict():
    """test update_nested_dict by checking outputs values"""
    # two simple dicts with different keys
    d = dict(d=1)
    v = dict(v=0)
    got = update_nested_dict(d, v)
    expected = dict(d=1, v=0)
    assert got == expected

    # two simple dicts with same key
    d = dict(d=1)
    v = dict(d=0)
    got = update_nested_dict(d, v)
    expected = dict(d=0)
    assert got == expected

    # dict with nested dict without common key
    d = dict(d=1)
    v = dict(v=dict(x=0))
    got = update_nested_dict(d, v)
    expected = dict(d=1, v=dict(x=0))
    assert got == expected

    # dict with nested dict with common key
    # fail because can not use dict to overwrite non dict values
    d = dict(v=1)
    v = dict(v=dict(x=0))
    with pytest.raises(TypeError) as err_info:
        update_nested_dict(d, v)
    assert "'int' object does not support item assignment" in str(err_info.value)

    # dict with nested dict with common key
    # pass because can use non dict to overwrite dict
    d = dict(v=dict(x=0))
    v = dict(v=1)
    got = update_nested_dict(d, v)
    expected = dict(v=1)
    assert got == expected

    # dict with nested dict with common key
    # overwrite a value
    d = dict(v=dict(x=0, y=1))
    v = dict(v=dict(x=1))
    got = update_nested_dict(d, v)
    expected = dict(v=dict(x=1, y=1))
    assert got == expected

    # dict with nested dict with common key
    # add a value
    d = dict(v=dict(x=0, y=1))
    v = dict(v=dict(z=1))
    got = update_nested_dict(d, v)
    expected = dict(v=dict(x=0, y=1, z=1))
    assert got == expected


class TestLoadConfigs:
    def test_single_config(self):
        with open("config/unpaired_labeled_ddf.yaml") as file:
            expected = yaml.load(file, Loader=yaml.FullLoader)
        got = load_configs("config/unpaired_labeled_ddf.yaml")
        assert got == expected

    def test_multiple_configs(self):
        with open("config/unpaired_labeled_ddf.yaml") as file:
            expected = yaml.load(file, Loader=yaml.FullLoader)
        got = load_configs(
            config_path=[
                "config/test/ddf.yaml",
                "config/test/unpaired_nifti.yaml",
                "config/test/labeled.yaml",
            ]
        )
        assert got == expected

    def test_outdated_config(self):
        with open("demos/grouped_mr_heart/grouped_mr_heart.yaml") as file:
            expected = yaml.load(file, Loader=yaml.FullLoader)
        got = load_configs("config/test/grouped_mr_heart_v011.yaml")
        assert got == expected
        updated_file_path = "config/test/updated_grouped_mr_heart_v011.yaml"
        assert os.path.isfile(updated_file_path)
        os.remove(updated_file_path)


def test_save():
    """test save by check error and existance of file"""
    # default file name
    with TempDirectory() as tempdir:
        save(config=dict(x=1), out_dir=tempdir.path)
        assert os.path.exists(os.path.join(tempdir.path, "config.yaml"))

    # custom file name
    with TempDirectory() as tempdir:
        save(config=dict(x=1), out_dir=tempdir.path, filename="test.yaml")
        assert os.path.exists(os.path.join(tempdir.path, "test.yaml"))

    # non yaml filename
    with TempDirectory() as tempdir:
        with pytest.raises(AssertionError):
            save(config=dict(x=1), out_dir=tempdir.path, filename="test.txt")


def test_config_sanity_check(caplog):
    """
    Test config_sanity_check by check error messages

    :param caplog: used to check warning message.
    """

    # unknown data type
    with pytest.raises(ValueError) as err_info:
        config_sanity_check(config=dict(dataset=dict(type="type")))
    assert "data type must be paired / unpaired / grouped" in str(err_info.value)

    # unknown data format
    with pytest.raises(ValueError) as err_info:
        config_sanity_check(config=dict(dataset=dict(type="paired", format="format")))
    assert "data format must be nifti / h5" in str(err_info.value)

    # dir is not in data_config
    with pytest.raises(AssertionError):
        config_sanity_check(config=dict(dataset=dict(type="paired", format="h5")))

    # dir doesn't have train/valid/test
    with pytest.raises(AssertionError):
        config_sanity_check(
            config=dict(dataset=dict(type="paired", format="h5", dir=dict()))
        )

    # train/valid/test of dir is not string or list of string
    with pytest.raises(ValueError) as err_info:
        config_sanity_check(
            config=dict(
                dataset=dict(
                    type="paired",
                    format="h5",
                    dir=dict(train=1, valid=None, test=None),
                    labeled=True,
                ),
                train=dict(model=dict(method="ddf")),
            )
        )
    assert "data_dir for mode train must be string or list of strings" in str(
        err_info.value
    )

    # use unlabeled data for conditional model
    with pytest.raises(ValueError) as err_info:
        config_sanity_check(
            config=dict(
                dataset=dict(
                    type="paired",
                    format="h5",
                    dir=dict(train=None, valid=None, test=None),
                    labeled=False,
                ),
                train=dict(
                    method="conditional",
                    loss=dict(),
                    preprocess=dict(),
                    optimizer=dict(name="Adam"),
                ),
            )
        )
    assert "For conditional model, data have to be labeled, got unlabeled data." in str(
        err_info.value
    )

    # check warnings
    # train/valid/test of dir is None
    # all loss weight <= 0
    caplog.clear()  # clear previous log
    config_sanity_check(
        config=dict(
            dataset=dict(
                type="paired",
                format="h5",
                dir=dict(train=None, valid=None, test=None),
                labeled=False,
            ),
            train=dict(
                method="ddf",
                loss=dict(
                    image=dict(name="lncc", weight=0.0),
                    label=dict(name="ssd", weight=0.0),
                    regularization=dict(name="bending", weight=0.0),
                ),
                preprocess=dict(),
                optimizer=dict(name="Adam"),
            ),
        )
    )
    # warning messages can be detected together
    assert "Data directory for train is not defined." in caplog.text
    assert "Data directory for valid is not defined." in caplog.text
    assert "Data directory for test is not defined." in caplog.text
