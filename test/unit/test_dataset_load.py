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
        """
        Test the data loader can be successfully built.

        :param data_type: name of data loader for registry
        :param format: name of file loader for registry
        """
        # single paired data loader
        config = load_yaml(f"config/test/{data_type}_{format}.yaml")
        got = load.get_data_loader(data_config=config["dataset"], split="train")
        expected = REGISTRY.get(category=DATA_LOADER_CLASS, key=data_type)
        assert isinstance(got, expected)  # type: ignore

    def test_multi_dir_data_loader(self):
        """unpaired data loader with multiple dirs"""
        config = load_yaml("config/test/unpaired_nifti_multi_dirs.yaml")
        got = load.get_data_loader(data_config=config["dataset"], split="train")
        assert isinstance(got, UnpairedDataLoader)

    @pytest.mark.parametrize("path", ["", None])
    def test_empty_path(self, path: Optional[str]):
        """
        Test return without data path.

        :param path: training data path to be used
        """
        config = load_yaml("config/test/paired_nifti.yaml")
        config["dataset"]["train"]["dir"] = path
        got = load.get_data_loader(data_config=config["dataset"], split="train")
        assert got is None

    @pytest.mark.parametrize("split", ["train", "valid", "test"])
    def test_empty_config(self, split: str):
        """
        Test return without data path for the split.

        :param split: train or valid or test
        """
        config = load_yaml("config/test/paired_nifti.yaml")
        config["dataset"].pop(split)
        got = load.get_data_loader(data_config=config["dataset"], split=split)
        assert got is None

    @pytest.mark.parametrize(
        "path", ["config/test/paired_nifti.yaml", "config/test/paired_nifti"]
    )
    def test_dir_err(self, path: Optional[str]):
        """
        Check the error is raised when the path is wrong.

        :param path: training data path to be used
        """
        config = load_yaml("config/test/paired_nifti.yaml")
        config["dataset"]["train"]["dir"] = path
        with pytest.raises(ValueError) as err_info:
            load.get_data_loader(data_config=config["dataset"], split="train")
        assert "is not a directory or does not exist" in str(err_info.value)

    def test_mode_err(self):
        """Check the error is raised when the split is wrong."""
        config = load_yaml("config/test/paired_nifti.yaml")
        with pytest.raises(ValueError) as err_info:
            load.get_data_loader(data_config=config["dataset"], split="example")
        assert "split must be one of ['train', 'valid', 'test']" in str(err_info.value)
