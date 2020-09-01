"""
Unit test for unpaired unlabeled data
"""

from test.output.util import train_and_predict_with_config


def test_unpaired_unlabeled():
    train_and_predict_with_config(
        test_name="unpaired_unlabeled_nifti",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/unpaired_nifti.yaml",
            "config/test/unlabeled.yaml",
        ],
    )

    train_and_predict_with_config(
        test_name="unpaired_unlabeled_h5",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/unpaired_h5.yaml",
            "config/test/unlabeled.yaml",
        ],
    )
