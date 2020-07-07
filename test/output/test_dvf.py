"""
Unit test for special config settings
"""

from deepreg.util import train_and_predict_with_config


def test_dvf():
    # the model outputs dvf
    train_and_predict_with_config(
        test_name="unpaired_unlabeled_nifti_dvf",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/dvf.yaml",
            "deepreg/config/test/unpaired_nifti.yaml",
        ],
    )
