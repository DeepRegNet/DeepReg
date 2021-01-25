# coding=utf-8

"""
Tests for deepreg/dataset/load.py in pytest style
"""
from typing import Optional

import pytest
import yaml

import deepreg.dataset.load as load
from deepreg.dataset.loader.unpaired_loader import UnpairedDataLoader
from deepreg.registry import DATA_LOADER_CLASS, REGISTRY


def load_yaml(file_path: str) -> dict:
    """
    load the yaml file and return a dictionary

    :param file_path: path of the yaml file.
    """
    assert file_path.endswith(".yaml")
    with open(file_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


class TestGetDataLoader:
    @pytest.mark.parametrize("data_type", ["paired", "unpaired", "grouped"])
    @pytest.mark.parametrize("format", ["nifti", "h5"])
    def test_data_loader(self, data_type: str, format: str):
        # single paired data loader
        config = load_yaml(f"config/test/{data_type}_{format}.yaml")
        got = load.get_data_loader(data_config=config["dataset"], mode="train")
        expected = REGISTRY.get(category=DATA_LOADER_CLASS, key=data_type)
        assert isinstance(got, expected)

    def test_multi_dir_data_loader(self):
        """unpaired data loader with multiple dirs"""
        config = load_yaml("config/test/unpaired_nifti_multi_dirs.yaml")
        got = load.get_data_loader(data_config=config["dataset"], mode="train")
        assert isinstance(got, UnpairedDataLoader)

    @pytest.mark.parametrize("path", ["", None])
    def test_empty_path(self, path: Optional[str]):
        """Test return without data path"""
        config = load_yaml("config/test/paired_nifti.yaml")
        config["dataset"]["dir"]["train"] = path
        got = load.get_data_loader(data_config=config["dataset"], mode="train")
        assert got is None

    @pytest.mark.parametrize("mode", ["train", "valid", "test"])
    def test_empty_config(self, mode: str):
        config = load_yaml("config/test/paired_nifti.yaml")
        config["dataset"]["dir"].pop(mode)
        got = load.get_data_loader(data_config=config["dataset"], mode=mode)
        assert got is None

    @pytest.mark.parametrize(
        "path", ["config/test/paired_nifti.yaml", "config/test/paired_nifti"]
    )
    def test_dir_err(self, path: Optional[str]):
        config = load_yaml("config/test/paired_nifti.yaml")
        config["dataset"]["dir"]["train"] = path
        with pytest.raises(ValueError) as err_info:
            load.get_data_loader(data_config=config["dataset"], mode="train")
        assert "is not a directory or does not exist" in str(err_info.value)

    def test_mode_err(self):
        config = load_yaml("config/test/paired_nifti.yaml")
        with pytest.raises(AssertionError) as err_info:
            load.get_data_loader(data_config=config["dataset"], mode="example")
        assert "mode must be one of train/valid/test" in str(err_info.value)
