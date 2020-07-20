"""
Help functions for output unit tests.
"""

import os

import tensorflow as tf

from deepreg.predict import predict
from deepreg.train import train


def train_and_predict_with_config(test_name, config_path):
    """
    Function that helps unit test whether a session with a given config
    runs the train and predict functions
    """
    tf.get_logger().setLevel(3)
    tf.keras.backend.clear_session()  # needed for pytest, otherwise model output name will change
    gpu = ""
    gpu_allow_growth = False
    ckpt_path = ""
    log_dir = "pytest_train_" + test_name
    train(
        gpu=gpu,
        config_path=config_path,
        gpu_allow_growth=gpu_allow_growth,
        ckpt_path=ckpt_path,
        log_dir=log_dir,
    )
    ckpt_path = os.path.join("logs", log_dir, "save", "weights-epoch2.ckpt")
    log_dir = "pytest_predict_" + test_name
    predict(
        gpu=gpu,
        gpu_allow_growth=gpu_allow_growth,
        ckpt_path=ckpt_path,
        mode="test",
        batch_size=1,
        log_dir=log_dir,
        sample_label="all",
        config_path="",
    )
