# coding=utf-8

"""
Tests for deepreg/train.py
pytest style
"""

import os

import pytest
import tensorflow as tf

from deepreg.predict import main as predict_main
from deepreg.train import build_callbacks, build_config
from deepreg.train import main as train_main

log_root = "logs"


class TestBuildConfig:
    # in the config, epochs = save_period = 2
    config_path = "config/unpaired_labeled_ddf.yaml"
    log_dir = "test_build_config"

    @pytest.mark.parametrize("ckpt_path", ["", "example.ckpt"])
    def test_ckpt_path(self, ckpt_path):
        # check the code can pass

        got_config, got_log_dir = build_config(
            config_path=self.config_path,
            log_root=log_root,
            log_dir=self.log_dir,
            ckpt_path=ckpt_path,
        )
        assert isinstance(got_config, dict)
        assert got_log_dir == os.path.join(log_root, self.log_dir)

    def test_ckpt_path_err(self):
        # checkpoint path ends with h5
        with pytest.raises(ValueError) as err_info:
            build_config(
                config_path=self.config_path,
                log_root=log_root,
                log_dir=self.log_dir,
                ckpt_path="example.h5",
            )
        assert "checkpoint path should end with .ckpt" in str(err_info.value)

    @pytest.mark.parametrize(
        "max_epochs, expected_epochs, expected_save_period", [(-1, 2, 2), (3, 3, 2)]
    )
    def test_max_epochs(self, max_epochs, expected_epochs, expected_save_period):
        got_config, _ = build_config(
            config_path=self.config_path,
            log_root=log_root,
            log_dir=self.log_dir,
            ckpt_path="",
            max_epochs=max_epochs,
        )
        assert got_config["train"]["epochs"] == expected_epochs
        assert got_config["train"]["save_period"] == expected_save_period


def test_build_callbacks():
    """
    Test build_callbacks by checking the output types
    """
    log_dir = "test_build_callbacks"
    histogram_freq = save_preiod = 1
    callbacks = build_callbacks(
        log_dir=log_dir, histogram_freq=histogram_freq, save_period=save_preiod
    )

    for callback in callbacks:
        assert isinstance(callback, tf.keras.callbacks.Callback)


def test_train_and_predict():
    """Covered by test_train_and_predict_main"""
    pass


def test_train_and_predict_main():
    """
    Test main in train and predict by checking it can run.
    """
    train_main(
        args=[
            "--gpu",
            "",
            "--log_dir",
            "test_train",
            "--config_path",
            "config/unpaired_labeled_ddf.yaml",
        ]
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
            "logs/test_train/save/weights-epoch2.ckpt",
            "--mode",
            "test",
            "--log_dir",
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
