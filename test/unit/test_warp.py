import os

from deepreg.warp import main


def test_warp():
    """Covered by test_main"""
    pass


def test_main():
    """
    Test main by checking the output file existance
    """
    image_path = "./data/test/nifti/unit_test/moving_image.nii.gz"
    ddf_path = "./data/test/nifti/unit_test/ddf.nii.gz"

    # custom output path with correct suffix
    out_path = "logs/test_warp/out.nii.gz"
    main(args=["--image", image_path, "--ddf", ddf_path, "--out", out_path])
    assert os.path.isfile(out_path)
    os.remove(out_path)

    # custom output path without correct suffix
    out_path = "logs/test_warp/out.h5"
    main(args=["--image", image_path, "--ddf", ddf_path, "--out", out_path])
    out_path = "logs/test_warp/warped.nii.gz"
    assert os.path.isfile(out_path)
    os.remove(out_path)

    # custom output path without correct suffix
    out_path = "logs/test_warp/"
    main(args=["--image", image_path, "--ddf", ddf_path, "--out", out_path])
    out_path = "logs/test_warp/warped.nii.gz"
    assert os.path.isfile(out_path)
    os.remove(out_path)

    # custom output path without correct suffix
    out_path = "logs/test_warp"
    main(args=["--image", image_path, "--ddf", ddf_path, "--out", out_path])
    out_path = "logs/warped.nii.gz"
    assert os.path.isfile(out_path)
    os.remove(out_path)

    # custom output path
    out_path = "warped.nii.gz"
    main(args=["--image", image_path, "--ddf", ddf_path, "--out", ""])
    assert os.path.isfile(out_path)
    os.remove(out_path)
