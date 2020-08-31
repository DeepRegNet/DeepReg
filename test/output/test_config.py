"""
Unit test for special config settings
"""

from test.output.util import train_and_predict_with_config


def test_unpaired_unlabeled():
    # the training set has multiple folders
    train_and_predict_with_config(
        test_name="unpaired_unlabeled_nifti_multi_dirs",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/unpaired_nifti_multi_dirs.yaml",
            "deepreg/config/test/unlabeled.yaml",
        ],
    )

    # data dir path for validation set is not provided
    train_and_predict_with_config(
        test_name="unpaired_unlabeled_nifti_no_valid",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/unpaired_nifti_no_valid.yaml",
            "deepreg/config/test/unlabeled.yaml",
        ],
    )
