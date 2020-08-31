"""
Unit test for grouped labeled data
"""
from test.output.util import train_and_predict_with_config


def test_grouped_labeled():
    train_and_predict_with_config(
        test_name="grouped_labeled_ddf",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/grouped_nifti.yaml",
            "deepreg/config/test/labeled.yaml",
        ],
    )

    train_and_predict_with_config(
        test_name="grouped_labeled_h5",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/grouped_h5.yaml",
            "deepreg/config/test/labeled.yaml",
        ],
    )
