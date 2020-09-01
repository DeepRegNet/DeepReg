"""
Unit test for grouped unlabeled data
"""

from test.output.util import train_and_predict_with_config


def test_grouped_unlabeled():
    train_and_predict_with_config(
        test_name="grouped_unlabeled_nifti",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/grouped_nifti.yaml",
            "config/test/unlabeled.yaml",
        ],
    )

    train_and_predict_with_config(
        test_name="grouped_unlabeled_h5",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/grouped_h5.yaml",
            "config/test/unlabeled.yaml",
        ],
    )
