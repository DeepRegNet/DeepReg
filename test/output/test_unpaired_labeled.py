"""
Unit test for unpaired labeled data
"""

from test.output.util import train_and_predict_with_config


def test_unpaired_labeled():
    # the nifti case is included in test_train.py:test_train

    # h5 case
    train_and_predict_with_config(
        test_name="unpaired_labeled_h5",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/unpaired_h5.yaml",
            "config/test/labeled.yaml",
        ],
    )
