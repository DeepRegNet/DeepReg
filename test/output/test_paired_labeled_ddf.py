"""
Unit test for paired labeled data
"""

from deepreg.util import train_and_predict_with_config


def test_paired_labeled_ddf():
    train_and_predict_with_config(
        test_name="paired_labeled_ddf",
        config_path="deepreg/config/paired_labeled_ddf.yaml",
    )

    train_and_predict_with_config(
        test_name="paired_labeled_h5",
        config_path="deepreg/config/h5_config/paired_labeled_h5.yaml",
    )
