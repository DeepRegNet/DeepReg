"""
Unit test for paired unlabeled data
"""
from test.output.util import train_and_predict_with_config


def test_paired_unlabeled():
    train_and_predict_with_config(
        test_name="paired_unlabeled_nifti",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/paired_nifti.yaml",
            "config/test/unlabeled.yaml",
        ],
    )

    train_and_predict_with_config(
        test_name="paired_unlabeled_h5",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/paired_h5.yaml",
            "config/test/unlabeled.yaml",
        ],
    )
