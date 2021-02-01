# coding=utf-8

"""
Tests for deepreg/train.py
pytest style
"""

import os
import shutil

from deepreg.predict import build_config, build_pair_output_path

log_root = "logs"


def test_build_pair_output_path():
    """
    Test build_config for labeled and unlabeled cases
    """

    save_dir = "logs/save_dir_example"

    # labeled
    got = build_pair_output_path(indices=[1, 2, 0], save_dir=save_dir)
    expected = (
        "logs/save_dir_example/pair_1_2",
        "logs/save_dir_example/pair_1_2/label_0",
    )
    assert got == expected
    assert os.path.exists(got[0])
    assert os.path.exists(got[1])
    shutil.rmtree(got[0])

    # unlabeled
    got = build_pair_output_path(indices=[1, 2, -1], save_dir=save_dir)
    expected = ("logs/save_dir_example/pair_1_2", "logs/save_dir_example/pair_1_2")
    assert got == expected
    assert os.path.exists(got[0])
    assert os.path.exists(got[1])
    shutil.rmtree(got[0])


def test_build_config():
    """
    Test build_config and check log_dir setting and checkpoint path verification
    """
    config_path = "config/unpaired_labeled_ddf.yaml"
    log_dir = "test_build_config"

    # TODO checkpoint path empty

    # checkpoint path ends with ckpt
    got_config, got_log_dir, _ = build_config(
        config_path=config_path,
        log_root=log_root,
        log_dir=log_dir,
        ckpt_path="example.ckpt",
    )
    assert isinstance(got_config, dict)
    assert got_log_dir == os.path.join("logs", log_dir)


def test_predict_on_dataset():
    # predict_on_dataset is tested in test_train/test_train_and_predict
    pass
