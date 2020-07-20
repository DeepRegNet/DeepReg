"""
Unit test for unpaired labeled data
"""

from deepreg.util import train_and_predict_with_config


def test_unpaired_labeled():
    # comment out the following test as it is included in test_train.py:test_train
    # train_and_predict_with_config(
    #     test_name="unpaired_labeled_nifti",
    #     config_path=[
    #         "deepreg/config/test/ddf.yaml",
    #         "deepreg/config/test/unpaired_nifti.yaml",
    #         "deepreg/config/test/labeled.yaml",
    #     ],
    # )

    train_and_predict_with_config(
        test_name="unpaired_labeled_h5",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/unpaired_h5.yaml",
            "deepreg/config/test/labeled.yaml",
        ],
    )
