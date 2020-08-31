"""
Unit test for paired labeled data
"""

from test.output.util import train_and_predict_with_config


def test_paired_labeled_ddf():
    train_and_predict_with_config(
        test_name="paired_labeled_nifti",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/paired_nifti.yaml",
            "deepreg/config/test/labeled.yaml",
        ],
    )

    train_and_predict_with_config(
        test_name="paired_labeled_h5",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/paired_h5.yaml",
            "deepreg/config/test/labeled.yaml",
        ],
    )
