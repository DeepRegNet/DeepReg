# coding=utf-8

"""
Tests for deepreg/train.py
pytest style
"""

import os
import shutil

import pytest

from deepreg.predict import main as predict_main
from deepreg.train import build_config
from deepreg.train import main as train_main


class TestBuildConfig:
    # in the config, epochs = save_period = 2
    config_path = "config/unpaired_labeled_ddf.yaml"
    exp_name = "test_build_config"
    log_dir = "logs"

    @pytest.mark.parametrize("ckpt_path", ["", "example.ckpt"])
    def test_ckpt_path(self, ckpt_path):
        # check the code can pass

        got_config, got_log_dir, _ = build_config(
            config_path=self.config_path,
            log_dir=self.log_dir,
            exp_name=self.exp_name,
            ckpt_path=ckpt_path,
        )
        assert isinstance(got_config, dict)
        assert got_log_dir == os.path.join(self.log_dir, self.exp_name)

    @pytest.mark.parametrize(
        "max_epochs, expected_epochs, expected_save_period", [(-1, 2, 2), (3, 3, 2)]
    )
    def test_max_epochs(self, max_epochs, expected_epochs, expected_save_period):
        got_config, _, _ = build_config(
            config_path=self.config_path,
            log_dir=self.log_dir,
            exp_name=self.exp_name,
            ckpt_path="",
            max_epochs=max_epochs,
        )
        assert got_config["train"]["epochs"] == expected_epochs
        assert got_config["train"]["save_period"] == expected_save_period


@pytest.mark.parametrize(
    "config_paths",
    [
        ["config/unpaired_labeled_ddf.yaml"],
        ["config/unpaired_labeled_ddf.yaml", "config/test/affine.yaml"],
    ],
)
def test_train_and_predict_main(config_paths):
    """
    Test main in train and predict by checking it can run.

    :param config_paths: list of file paths for configuration.
    """
    train_main(
        args=[
            "--gpu",
            "",
            "--exp_name",
            "test_train",
            "--config_path",
        ]
        + config_paths
    )

    # check output folders
    assert os.path.isdir("logs/test_train/save")
    assert os.path.isdir("logs/test_train/train")
    assert os.path.isdir("logs/test_train/validation")
    assert os.path.isfile("logs/test_train/config.yaml")

    predict_main(
        args=[
            "--gpu",
            "",
            "--ckpt_path",
            "logs/test_train/save/ckpt-2",
            "--split",
            "test",
            "--exp_name",
            "test_predict",
            "--save_nifti",
            "--save_png",
        ]
    )

    # check output folders
    assert os.path.isdir("logs/test_predict/test/pair_0_1/label_0")
    assert os.path.isdir("logs/test_predict/test/pair_0_1/label_1")
    assert os.path.isdir("logs/test_predict/test/pair_0_1/label_2")
    assert os.path.isfile("logs/test_predict/test/metrics.csv")
    assert os.path.isfile("logs/test_predict/test/metrics_stats_per_label.csv")
    assert os.path.isfile("logs/test_predict/test/metrics_stats_overall.csv")

    shutil.rmtree("logs/test_train")
    shutil.rmtree("logs/test_predict")
