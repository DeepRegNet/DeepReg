# coding=utf-8

"""
Tests for deepreg/dataset/load.py in pytest style
"""
import pytest
import yaml

import deepreg.dataset.load as load
from deepreg.dataset.loader.grouped_loader import GroupedDataLoader
from deepreg.dataset.loader.interface import ConcatenatedDataLoader
from deepreg.dataset.loader.nifti_loader import NiftiFileLoader
from deepreg.dataset.loader.paired_loader import PairedDataLoader
from deepreg.dataset.loader.unpaired_loader import UnpairedDataLoader


def load_yaml(file_path: str) -> dict:
    """
    load the yaml file and return a dictionary
    """
    assert file_path.endswith(".yaml")
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def test_get_data_loader():
    """
    Test for get_data_loader to make sure it get correct data loader and raise correct errors
    """

    # single paired data loader
    config = load_yaml("deepreg/config/test/paired_nifti.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, PairedDataLoader)

    config = load_yaml("deepreg/config/test/paired_h5.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, PairedDataLoader)

    # single unpaired data loader
    config = load_yaml("deepreg/config/test/unpaired_nifti.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, UnpairedDataLoader)

    config = load_yaml("deepreg/config/test/unpaired_h5.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, UnpairedDataLoader)

    # single grouped data loader
    config = load_yaml("deepreg/config/test/grouped_nifti.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, GroupedDataLoader)

    config = load_yaml("deepreg/config/test/grouped_h5.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, GroupedDataLoader)

    # empty data loader
    config = load_yaml("deepreg/config/test/paired_nifti.yaml")
    config["dataset"]["dir"]["train"] = ""
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert got is None

    config = load_yaml("deepreg/config/test/paired_nifti.yaml")
    config["dataset"]["dir"]["train"] = None
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert got is None

    # concatenated unpaired data loader
    config = load_yaml("deepreg/config/test/unpaired_nifti_multi_dirs.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, ConcatenatedDataLoader)

    # check not a directory error
    config = load_yaml("deepreg/config/test/paired_nifti.yaml")
    config["dataset"]["dir"]["train"] += ".h5"
    with pytest.raises(ValueError) as execinfo:
        load.get_data_loader(data_config=config["dataset"], mode="train")
    msg = " ".join(execinfo.value.args[0].split())
    assert "is not a directory or does not exist" in msg

    # check directory not existed error
    config = load_yaml("deepreg/config/test/paired_nifti.yaml")
    config["dataset"]["dir"]["train"] = "/this_should_not_existed"
    with pytest.raises(ValueError) as execinfo:
        load.get_data_loader(data_config=config["dataset"], mode="train")
    msg = " ".join(execinfo.value.args[0].split())
    assert "is not a directory or does not exist" in msg


def test_get_single_data_loader():
    """
    Test for get_single_data_loader to make sure it get correct data loader and raise correct errors
    Mainly based on nifti file loader
    """
    common_args = dict(
        file_loader=NiftiFileLoader, labeled=True, sample_label="sample", seed=0
    )

    # single paired data loader
    config = load_yaml("deepreg/config/test/paired_nifti.yaml")
    got = load.get_single_data_loader(
        data_type=config["dataset"]["type"],
        data_config=config["dataset"],
        common_args=common_args,
        data_dir_path=config["dataset"]["dir"]["train"],
    )
    assert isinstance(got, PairedDataLoader)

    # single unpaired data loader
    config = load_yaml("deepreg/config/test/unpaired_nifti.yaml")
    got = load.get_single_data_loader(
        data_type=config["dataset"]["type"],
        data_config=config["dataset"],
        common_args=common_args,
        data_dir_path=config["dataset"]["dir"]["train"],
    )
    assert isinstance(got, UnpairedDataLoader)

    # single grouped data loader
    config = load_yaml("deepreg/config/test/grouped_nifti.yaml")
    got = load.get_single_data_loader(
        data_type=config["dataset"]["type"],
        data_config=config["dataset"],
        common_args=common_args,
        data_dir_path=config["dataset"]["dir"]["train"],
    )
    assert isinstance(got, GroupedDataLoader)

    # not supported data loader
    config = load_yaml("deepreg/config/test/paired_nifti.yaml")
    with pytest.raises(ValueError) as execinfo:
        load.get_single_data_loader(
            data_type="NotSupported",
            data_config=config["dataset"],
            common_args=common_args,
            data_dir_path=config["dataset"]["dir"]["train"],
        )
    msg = " ".join(execinfo.value.args[0].split())
    assert "Unknown data format" in msg
