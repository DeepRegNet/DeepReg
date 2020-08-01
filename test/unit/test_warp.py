import os
import shutil

from deepreg.warp import warp


def test_warp():
    """
    Test warp by checking the output file existance
    """
    image_path = "./data/test/nifti/unit_test/moving_image.nii.gz"
    ddf_path = "./data/test/nifti/unit_test/ddf.nii.gz"

    # custom output path
    out_path = "logs/test_warp/out.nii.gz"
    warp(image_path=image_path, ddf_path=ddf_path, out_path=out_path)
    assert os.path.isfile(out_path)
    shutil.rmtree(os.path.dirname(out_path))

    # custom output path
    out_path = "warped.nii.gz"
    warp(image_path=image_path, ddf_path=ddf_path, out_path="")
    assert os.path.isfile(out_path)
    os.remove(out_path)
