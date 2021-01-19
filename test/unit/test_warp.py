import os

import numpy as np
import pytest

from deepreg.warp import main, shape_sanity_check

image_path = "./data/test/nifti/unit_test/moving_image.nii.gz"
ddf_path = "./data/test/nifti/unit_test/ddf.nii.gz"


@pytest.mark.parametrize(
    ("out_path", "expected_path"),
    [
        ("logs/test_warp/out.nii.gz", "logs/test_warp/out.nii.gz"),
        ("logs/test_warp/out.h5", "logs/test_warp/warped.nii.gz"),
        ("logs/test_warp/", "logs/test_warp/warped.nii.gz"),
        ("", "warped.nii.gz"),
    ],
)
def test_main(out_path: str, expected_path: str):
    main(args=["--image", image_path, "--ddf", ddf_path, "--out", out_path])
    assert os.path.isfile(expected_path)
    os.remove(expected_path)


class TestShapeSanityCheck:
    @pytest.mark.parametrize(
        ("image_shape", "ddf_shape"),
        [
            ((2, 3, 4), (2, 3, 4, 3)),
            ((2, 3, 4, 1), (2, 3, 4, 3)),
            ((2, 3, 4, 3), (2, 3, 4, 3)),
        ],
    )
    def test_pass(self, image_shape: tuple, ddf_shape: tuple):
        image = np.ones(image_shape)
        ddf = np.ones(ddf_shape)
        shape_sanity_check(image=image, ddf=ddf)

    @pytest.mark.parametrize(
        ("image_shape", "ddf_shape", "err_msg"),
        [
            (
                (
                    2,
                    3,
                ),
                (2, 3, 4, 3),
                "image shape must be (m_dim1, m_dim2, m_dim3)",
            ),
            ((2, 3, 4), (2, 3, 4, 4), "ddf shape must be (f_dim1, f_dim2, f_dim3, 3)"),
        ],
    )
    def test_error(self, image_shape: tuple, ddf_shape: tuple, err_msg):
        image = np.ones(image_shape)
        ddf = np.ones(ddf_shape)
        with pytest.raises(ValueError) as err_info:
            shape_sanity_check(image=image, ddf=ddf)
        assert err_msg in str(err_info.value)
