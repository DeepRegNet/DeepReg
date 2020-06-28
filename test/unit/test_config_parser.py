"""
Tests functionalities in deepreg/config/parser.py
"""
import unittest

from deepreg.config.parser import load_configs


class TestConfigParser(unittest.TestCase):
    def test_multi_configs(self):
        expected = load_configs(
            config_path="deepreg/config/unpaired_unlabeled_ddf.yaml"
        )
        got = load_configs(
            config_path=[
                "deepreg/config/test/ddf.yaml",
                "deepreg/config/test/unpaired_nifti.yaml",
                "deepreg/config/test/unlabeled.yaml",
            ]
        )
        self.assertEqual(got, expected)
