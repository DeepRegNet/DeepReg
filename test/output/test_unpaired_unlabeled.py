"""
Unit test for unpaired unlabeled data
"""

from deepreg.test_util import train_and_predict_with_config


def test_unpaired_unlabeled():
    train_and_predict_with_config(
        test_name="unpaired_unlabeled_nifti",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/unpaired_nifti.yaml",
            "deepreg/config/test/unlabeled.yaml",
        ],
    )

    train_and_predict_with_config(
        test_name="unpaired_unlabeled_h5",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/unpaired_h5.yaml",
            "deepreg/config/test/unlabeled.yaml",
        ],
    )
