import os

import tensorflow as tf

from deepreg.predict import predict
from deepreg.train import train

tf.get_logger().setLevel(3)


def train_and_predict_with_config(test_name, config_path):
    tf.keras.backend.clear_session()  # needed for pytest, otherwise model output name will change
    gpu = ""
    gpu_allow_growth = False
    ckpt_path = ""
    log_dir = os.path.join("logs", "test_" + test_name)
    train(gpu=gpu, config_path=config_path, gpu_allow_growth=gpu_allow_growth, ckpt_path=ckpt_path, log_dir=log_dir)
    ckpt_path = os.path.join(log_dir, "save", "weights-epoch2.ckpt")
    predict(gpu=gpu, gpu_allow_growth=gpu_allow_growth, ckpt_path=ckpt_path, mode="test",
            batch_size=1, log_dir=log_dir, sample_label="all")


class TestTrainAndPredict:
    def test_paired_labeled_ddf(self):
        train_and_predict_with_config(test_name="paired_labeled_ddf",
                                      config_path="deepreg/config/paired_labeled_ddf.yaml")

    def test_paired_unlabeled_ddf(self):
        train_and_predict_with_config(test_name="paired_unlabeled_ddf",
                                      config_path="deepreg/config/paired_unlabeled_ddf.yaml")

    def test_unpaired_labeled_ddf(self):
        train_and_predict_with_config(test_name="unpaired_labeled_ddf",
                                      config_path="deepreg/config/unpaired_labeled_ddf.yaml")

    def test_unpaired_unlabeled_ddf(self):
        train_and_predict_with_config(test_name="unpaired_unlabeled_ddf",
                                      config_path="deepreg/config/unpaired_unlabeled_ddf.yaml")

    def test_grouped_labeled_ddf(self):
        train_and_predict_with_config(test_name="grouped_labeled_ddf",
                                      config_path="deepreg/config/grouped_labeled_ddf.yaml")

    def test_grouped_unlabeled_ddf(self):
        train_and_predict_with_config(test_name="grouped_unlabeled_ddf",
                                      config_path="deepreg/config/grouped_unlabeled_ddf.yaml")

    def test_paired_labeled_h5(self):
        train_and_predict_with_config(test_name="paired_labeled_h5",
                                      config_path="deepreg/config/h5_config/paired_labeled_h5.yaml")

    def test_paired_unlabeled_h5(self):
        train_and_predict_with_config(test_name="paired_unlabeled_h5",
                                      config_path="deepreg/config/h5_config/paired_unlabeled_h5.yaml")

    def test_unpaired_labeled_h5(self):
        train_and_predict_with_config(test_name="unpaired_labeled_h5",
                                      config_path="deepreg/config/h5_config/unpaired_labeled_h5.yaml")

    def test_unpaired_unlabeled_h5(self):
        train_and_predict_with_config(test_name="unpaired_unlabeled_h5",
                                      config_path="deepreg/config/h5_config/unpaired_unlabeled_h5.yaml")

    def test_grouped_labeled_h5(self):
        train_and_predict_with_config(test_name="grouped_labeled_h5",
                                      config_path="deepreg/config/h5_config/grouped_labeled_h5.yaml")

    def test_grouped_unlabeled_h5(self):
        train_and_predict_with_config(test_name="grouped_unlabeled_h5",
                                      config_path="deepreg/config/h5_config/grouped_unlabeled_h5.yaml")
