"""
Tests functionality of the H5FileLoader
"""
import unittest

from deepreg.dataset.loader.h5_loader import H5FileLoader


class TestH5FileLoader(unittest.TestCase):
    def test_data_keys(self):
        """
        check if the data_keys are the same as expected for paired data
        TODO add test for unpaired and grouped
        :return: bool, True if test passed
        """
        dir_path = "data/test/h5/paired/test"
        name = "fixed_images"

        loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
        got = loader.data_keys
        expected = ["case000025.nii.gz"]
        self.assertEqual(got, expected)
