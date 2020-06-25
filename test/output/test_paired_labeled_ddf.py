"""
Unit test for paired labeled data
"""
import os
import tensorflow as tf
from deepreg.predict import predict
from deepreg.train import train

def test_paired_labeled_ddf():
    tf.get_logger().setLevel(3)
    def train_and_predict_with_config(test_name, config_path):
        """
        Function that helps unit test whether a session with a given config
        runs the train and predict functions
        """
        tf.keras.backend.clear_session()  # needed for pytest, otherwise model output name will change
        gpu = ""
        gpu_allow_growth = False
        ckpt_path = ""
        log_dir = os.path.join("logs", "test_" + test_name)
        train(gpu=gpu, config_path=config_path, gpu_allow_growth=gpu_allow_growth, ckpt_path=ckpt_path, log_dir=log_dir)
        ckpt_path = os.path.join(log_dir, "save", "weights-epoch2.ckpt")
        predict(gpu=gpu, gpu_allow_growth=gpu_allow_growth, ckpt_path=ckpt_path, mode="test",
                batch_size=1, log_dir=log_dir, sample_label="all")
    train_and_predict_with_config(test_name="paired_labeled_ddf",
                                    config_path="deepreg/config/paired_labeled_ddf.yaml")