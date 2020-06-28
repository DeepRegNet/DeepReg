"""
Unit test for paired unlabeled data
"""
from deepreg.util import train_and_predict_with_config


def test_paired_unlabeled():
    train_and_predict_with_config(
        test_name="paired_unlabeled_nifti",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/paired_nifti.yaml",
            "deepreg/config/test/unlabeled.yaml",
        ],
    )

    train_and_predict_with_config(
        test_name="paired_unlabeled_h5",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/paired_h5.yaml",
            "deepreg/config/test/unlabeled.yaml",
        ],
    )
