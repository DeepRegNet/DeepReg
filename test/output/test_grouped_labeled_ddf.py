"""
Unit test for grouped labeled data
"""
from deepreg.util import train_and_predict_with_config


def test_grouped_labeled_ddf():
    train_and_predict_with_config(
        test_name="grouped_labeled_ddf",
        config_path="deepreg/config/grouped_labeled_ddf.yaml",
    )

    train_and_predict_with_config(
        test_name="grouped_labeled_h5",
        config_path="deepreg/config/h5_config/grouped_labeled_h5.yaml",
    )
