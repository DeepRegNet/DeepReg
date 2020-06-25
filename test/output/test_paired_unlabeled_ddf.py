"""
Unit test for paired unlabeled data
"""
from deepreg.util import train_and_predict_with_config


def test_paired_unlabeled_ddf():
    train_and_predict_with_config(test_name="paired_unlabeled_ddf",
                                  config_path="deepreg/config/paired_unlabeled_ddf.yaml")

    train_and_predict_with_config(test_name="paired_unlabeled_h5",
                                  config_path="deepreg/config/h5_config/paired_unlabeled_h5.yaml")
