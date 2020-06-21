import os

import tensorflow as tf

from deepreg.predict import predict
from deepreg.train import train

tf.get_logger().setLevel(3)


def train_and_predict_with_config(test_name, config_path):
    gpu = ""
    gpu_allow_growth = False
    ckpt_path = ""
    log_dir = os.path.join("logs", "test_" + test_name)
    train(gpu=gpu, config_path=config_path, gpu_allow_growth=gpu_allow_growth, ckpt_path=ckpt_path, log_dir=log_dir)
    ckpt_path = os.path.join(log_dir, "save", "weights-epoch2.ckpt")
    predict(gpu=gpu, gpu_allow_growth=gpu_allow_growth, ckpt_path=ckpt_path, mode="test",
            batch_size=1, log_dir=log_dir, sample_label="all")


def test_train_and_predict():
    configs = [
        #("paired_unlabeled_ddf", "deepreg/config/paired_unlabeled_ddf.yaml"),
        #("paired_labeled_ddf", "deepreg/config/paired_labeled_ddf.yaml"),
        #("unpaired_unlabeled_ddf", "deepreg/config/unpaired_unlabeled_ddf.yaml"),
        #("unpaired_labeled_ddf", "deepreg/config/unpaired_labeled_ddf.yaml"),
        ("paired_labeled_h5", "deepreg/config/paired_labeled_h5.yaml"),
        #("paired_unlabeled_h5", "deepreg/config/paired_unlabeled_h5.yaml"),
        #("unpaired_labeled_h5", "deepreg/config/unpaired_labeled_h5.yaml"),
        #("unpaired_unlabeled_h5", "deepreg/config/unpaired_unlabeled_h5.yaml"),
    ]
    for test_name, config_path in configs:
        tf.keras.backend.clear_session()  # needed for pytest, otherwise model output name will change
        train_and_predict_with_config(test_name, config_path)
