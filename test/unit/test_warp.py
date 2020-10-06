import os

import pytest

from deepreg.warp import main


def test_warp():
    """Covered by test_main"""
    pass


class TestMain:
    image_path = "./data/test/nifti/unit_test/moving_image.nii.gz"
    ddf_path = "./data/test/nifti/unit_test/ddf.nii.gz"

    @pytest.mark.parametrize(
        "out_path,expected_path",
        [
            ["logs/test_warp/out.nii.gz", "logs/test_warp/out.nii.gz"],
            ["logs/test_warp/out.h5", "logs/test_warp/warped.nii.gz"],
            ["logs/test_warp/", "logs/test_warp/warped.nii.gz"],
            ["", "warped.nii.gz"],
        ],
    )
    def test_main(self, out_path: str, expected_path: str):
        """
        Integration test by checking the output file exists
        """
        main(
            args=["--image", self.image_path, "--ddf", self.ddf_path, "--out", out_path]
        )
        assert os.path.isfile(expected_path)
        os.remove(expected_path)
