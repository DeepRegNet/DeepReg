# coding=utf-8

"""
Tests for deepreg/dataset/load.py in pytest style
"""
import pytest
import yaml

import deepreg.dataset.load as load
from deepreg.dataset.loader.grouped_loader import GroupedDataLoader
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
    config = load_yaml("config/test/paired_nifti.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, PairedDataLoader)

    config = load_yaml("config/test/paired_h5.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, PairedDataLoader)

    # single unpaired data loader
    config = load_yaml("config/test/unpaired_nifti.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, UnpairedDataLoader)

    config = load_yaml("config/test/unpaired_h5.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, UnpairedDataLoader)

    # single grouped data loader
    config = load_yaml("config/test/grouped_nifti.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, GroupedDataLoader)

    config = load_yaml("config/test/grouped_h5.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, GroupedDataLoader)

    # empty data loader
    config = load_yaml("config/test/paired_nifti.yaml")
    config["dataset"]["dir"]["train"] = ""
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert got is None

    config = load_yaml("config/test/paired_nifti.yaml")
    config["dataset"]["dir"]["train"] = None
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert got is None

    # unpaired data loader with multiple dirs
    config = load_yaml("config/test/unpaired_nifti_multi_dirs.yaml")
    got = load.get_data_loader(data_config=config["dataset"], mode="train")
    assert isinstance(got, UnpairedDataLoader)

    # check not a directory error
    config = load_yaml("config/test/paired_nifti.yaml")
    config["dataset"]["dir"]["train"] += ".h5"
    with pytest.raises(ValueError) as err_info:
        load.get_data_loader(data_config=config["dataset"], mode="train")
    assert "is not a directory or does not exist" in str(err_info.value)

    # check directory not existed error
    config = load_yaml("config/test/paired_nifti.yaml")
    config["dataset"]["dir"]["train"] = "/this_should_not_existed"
    with pytest.raises(ValueError) as err_info:
        load.get_data_loader(data_config=config["dataset"], mode="train")
    assert "is not a directory or does not exist" in str(err_info.value)

    # check mode
    config = load_yaml("config/test/paired_nifti.yaml")
    with pytest.raises(AssertionError) as err_info:
        load.get_data_loader(data_config=config["dataset"], mode="example")
    assert "mode must be one of train/valid/test" in str(err_info.value)


def test_get_single_data_loader():
    """
    Test for get_single_data_loader to make sure it get correct data loader and raise correct errors
    Mainly based on nifti file loader
    """
    common_args = dict(
        file_loader=NiftiFileLoader, labeled=True, sample_label="sample", seed=0
    )

    # single paired data loader
    config = load_yaml("config/test/paired_nifti.yaml")
    got = load.get_single_data_loader(
        data_type=config["dataset"]["type"],
        data_config=config["dataset"],
        common_args=common_args,
        data_dir_paths=[config["dataset"]["dir"]["train"]],
    )
    assert isinstance(got, PairedDataLoader)

    # single unpaired data loader
    config = load_yaml("config/test/unpaired_nifti.yaml")
    got = load.get_single_data_loader(
        data_type=config["dataset"]["type"],
        data_config=config["dataset"],
        common_args=common_args,
        data_dir_paths=[config["dataset"]["dir"]["train"]],
    )
    assert isinstance(got, UnpairedDataLoader)

    # single grouped data loader
    config = load_yaml("config/test/grouped_nifti.yaml")
    got = load.get_single_data_loader(
        data_type=config["dataset"]["type"],
        data_config=config["dataset"],
        common_args=common_args,
        data_dir_paths=[config["dataset"]["dir"]["train"]],
    )
    assert isinstance(got, GroupedDataLoader)

    # not supported data loader
    config = load_yaml("config/test/paired_nifti.yaml")
    with pytest.raises(ValueError) as err_info:
        load.get_single_data_loader(
            data_type="NotSupported",
            data_config=config["dataset"],
            common_args=common_args,
            data_dir_paths=[config["dataset"]["dir"]["train"]],
        )
    assert "Unknown data format" in str(err_info.value)

    # wrong keys for paired loader
    config = load_yaml("config/test/paired_nifti.yaml")
    # delete correct keys and add wrong one
    config["dataset"].pop("moving_image_shape", None)
    config["dataset"].pop("fixed_image_shape", None)
    with pytest.raises(ValueError) as err_info:
        load.get_single_data_loader(
            data_type="paired",
            data_config=config["dataset"],
            common_args=common_args,
            data_dir_paths=[config["dataset"]["dir"]["train"]],
        )
    assert "Paired Loader requires 'moving_image_shape' and 'fixed_image_shape'" in str(
        err_info.value
    )

    # wrong keys for unpaired loader
    config = load_yaml("config/test/unpaired_nifti.yaml")
    # delete correct keys and add wrong one
    config["dataset"].pop("image_shape", None)
    with pytest.raises(ValueError) as err_info:
        load.get_single_data_loader(
            data_type="unpaired",
            data_config=config["dataset"],
            common_args=common_args,
            data_dir_paths=[config["dataset"]["dir"]["train"]],
        )
    assert "Unpaired Loader requires 'image_shape'" in str(err_info.value)

    # wrong keys for grouped loader
    config = load_yaml("config/test/unpaired_nifti.yaml")
    # delete correct keys and add wrong one
    config["dataset"].pop("intra_group_prob", None)
    with pytest.raises(ValueError) as err_info:
        load.get_single_data_loader(
            data_type="grouped",
            data_config=config["dataset"],
            common_args=common_args,
            data_dir_paths=[config["dataset"]["dir"]["train"]],
        )
    assert "Grouped Loader requires 'image_shape'" in str(err_info.value)
