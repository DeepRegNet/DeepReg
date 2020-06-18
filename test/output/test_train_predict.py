import os

import tensorflow as tf

from deepreg.predict import predict
from deepreg.train import train

tf.get_logger().setLevel(3)


def test_train_and_predict():
    configs = [
        ("paired_labeled_ddf", "deepreg/config/paired_labeled_ddf.yaml"),
        ("unpaired_labeled_ddf", "deepreg/config/unpaired_labeled_ddf.yaml"),
    ]
    gpu = ""
    gpu_allow_growth = False
    ckpt_path = ""
    for test_name, config_path in configs:
        log_dir = os.path.join("logs", "test_" + test_name)
        train(gpu=gpu, config_path=config_path, gpu_allow_growth=gpu_allow_growth, ckpt_path=ckpt_path, log_dir=log_dir)
        ckpt_path = os.path.join(log_dir, "save", "weights-epoch2.ckpt")
        predict(gpu=gpu, gpu_allow_growth=gpu_allow_growth, ckpt_path=ckpt_path, mode="test",
                batch_size=1, log_dir=log_dir, sample_label="all")
