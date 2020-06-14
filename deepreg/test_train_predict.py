import os

import tensorflow as tf

from deepreg.predict import predict
from deepreg.train import train

tf.get_logger().setLevel(3)


def test_train_and_predict():
    gpu = "0"
    config_path = "deepreg/config/mr_us_ddf.yaml"
    gpu_allow_growth = False
    ckpt_path = ""
    log = "test_train"
    train(gpu=gpu, config_path=config_path, gpu_allow_growth=gpu_allow_growth, ckpt_path=ckpt_path, log=log)
    ckpt_path = os.path.join("logs", log, "save", "weights-epoch2.ckpt")
    mode = "test"
    batch_size = 1
    sample_label = "all"
    predict(gpu=gpu, gpu_allow_growth=gpu_allow_growth, ckpt_path=ckpt_path, mode=mode,
            batch_size=batch_size, log=log, sample_label=sample_label)
