"""
Unit test for unpaired unlabeled data
"""

from deepreg.util import train_and_predict_with_config


def test_unpaired_unlabeled_ddf():
    train_and_predict_with_config(
        test_name="unpaired_unlabeled_ddf",
        config_path="deepreg/config/unpaired_unlabeled_ddf.yaml",
    )

    train_and_predict_with_config(
        test_name="unpaired_unlabeled_h5",
        config_path="deepreg/config/h5_config/unpaired_unlabeled_h5.yaml",
    )
