"""
Unit test for special config settings
"""

from test.output.util import train_and_predict_with_config


def test_unpaired_unlabeled():
    # the training set has multiple folders
    train_and_predict_with_config(
        test_name="unpaired_unlabeled_nifti_multi_dirs",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/unpaired_nifti_multi_dirs.yaml",
            "config/test/unlabeled.yaml",
        ],
    )

    # data dir path for validation set is not provided
    train_and_predict_with_config(
        test_name="unpaired_unlabeled_nifti_no_valid",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/unpaired_nifti_no_valid.yaml",
            "config/test/unlabeled.yaml",
        ],
    )
