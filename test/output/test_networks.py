"""
Unit test for special config settings
"""

from test.output.util import train_and_predict_with_config


def test_dvf_network():
    # the model outputs dvf
    train_and_predict_with_config(
        test_name="unpaired_labeled_nifti_dvf",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/dvf.yaml",
            "config/test/unpaired_nifti.yaml",
            "config/test/labeled.yaml",
        ],
    )


def test_conditional_network():
    # the model outputs predicted label
    train_and_predict_with_config(
        test_name="unpaired_labeled_nifti_conditional",
        config_path=[
            "config/test/ddf.yaml",
            "config/test/conditional.yaml",
            "config/test/unpaired_nifti.yaml",
            "config/test/labeled.yaml",
        ],
    )
