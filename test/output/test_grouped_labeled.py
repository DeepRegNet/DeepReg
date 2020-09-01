"""
Unit test for grouped labeled data
"""
from test.output.util import train_and_predict_with_config


def test_grouped_labeled():
    train_and_predict_with_config(
        test_name="grouped_labeled_ddf",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/grouped_nifti.yaml",
            "config/test/labeled.yaml",
        ],
    )

    train_and_predict_with_config(
        test_name="grouped_labeled_h5",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/grouped_h5.yaml",
            "config/test/labeled.yaml",
        ],
    )
