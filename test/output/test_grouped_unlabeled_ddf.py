"""
Unit test for grouped unlabeled data
"""

from deepreg.util import train_and_predict_with_config


def test_grouped_unlabeled_ddf():
    train_and_predict_with_config(
        test_name="grouped_unlabeled_ddf",
        config_path="deepreg/config/grouped_unlabeled_ddf.yaml",
    )

    train_and_predict_with_config(
        test_name="grouped_unlabeled_h5",
        config_path="deepreg/config/h5_config/grouped_unlabeled_h5.yaml",
    )
