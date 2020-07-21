# coding=utf-8

"""
Tests for deepreg/train.py
pytest style
"""

import os

import pytest
import tensorflow as tf

from deepreg.train import build_callbacks, build_config, train


def test_build_config():
    """
    Test build_config and check log_dir setting and checkpoint path verification
    """
    config_path = "deepreg/config/unpaired_labeled_ddf.yaml"
    log_dir = "test_build_config"

    # checkpoint path empty
    got_config, got_log_dir = build_config(
        config_path=config_path, log_dir=log_dir, ckpt_path=""
    )
    assert isinstance(got_config, dict)
    assert got_log_dir == os.path.join("logs", log_dir)

    # checkpoint path ends with ckpt
    got_config, got_log_dir = build_config(
        config_path=config_path, log_dir=log_dir, ckpt_path="example.ckpt"
    )
    assert isinstance(got_config, dict)
    assert got_log_dir == os.path.join("logs", log_dir)

    # checkpoint path ends with h5
    with pytest.raises(ValueError):
        build_config(config_path=config_path, log_dir=log_dir, ckpt_path="example.h5")


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


def test_train():
    """
    Test train by checking it can run.
    """
    gpu = ""
    config_path = "deepreg/config/unpaired_labeled_ddf.yaml"
    gpu_allow_growth = False
    ckpt_path = ""
    log_dir = "test_train"
    train(
        gpu=gpu,
        config_path=config_path,
        gpu_allow_growth=gpu_allow_growth,
        ckpt_path=ckpt_path,
        log_dir=log_dir,
    )
