"""
Unit test for special config settings
"""

from deepreg.test_util import train_and_predict_with_config


def test_dvf_network():
    # the model outputs dvf
    train_and_predict_with_config(
        test_name="unpaired_labeled_nifti_dvf",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/dvf.yaml",
            "deepreg/config/test/unpaired_nifti.yaml",
            "deepreg/config/test/labeled.yaml",
        ],
    )


def test_conditional_network():
    # the model outputs predicted label
    train_and_predict_with_config(
        test_name="unpaired_labeled_nifti_conditional",
        config_path=[
            "deepreg/config/test/ddf.yaml",
            "deepreg/config/test/conditional.yaml",
            "deepreg/config/test/unpaired_nifti.yaml",
            "deepreg/config/test/labeled.yaml",
        ],
    )
