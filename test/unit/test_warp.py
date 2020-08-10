import os

from deepreg.warp import main, warp


def test_warp():
    """
    Test warp by checking the output file existance
    """
    image_path = "./data/test/nifti/unit_test/moving_image.nii.gz"
    ddf_path = "./data/test/nifti/unit_test/ddf.nii.gz"

    # custom output path with correct suffix
    out_path = "logs/test_warp/out.nii.gz"
    warp(image_path=image_path, ddf_path=ddf_path, out_path=out_path)
    assert os.path.isfile(out_path)
    os.remove(out_path)

    # custom output path without correct suffix
    out_path = "logs/test_warp/out.h5"
    warp(image_path=image_path, ddf_path=ddf_path, out_path=out_path)
    out_path = "logs/test_warp/warped.nii.gz"
    assert os.path.isfile(out_path)
    os.remove(out_path)

    # custom output path without correct suffix
    out_path = "logs/test_warp/"
    warp(image_path=image_path, ddf_path=ddf_path, out_path=out_path)
    out_path = "logs/test_warp/warped.nii.gz"
    assert os.path.isfile(out_path)
    os.remove(out_path)

    # custom output path without correct suffix
    out_path = "logs/test_warp"
    warp(image_path=image_path, ddf_path=ddf_path, out_path=out_path)
    out_path = "logs/warped.nii.gz"
    assert os.path.isfile(out_path)
    os.remove(out_path)

    # custom output path
    out_path = "warped.nii.gz"
    warp(image_path=image_path, ddf_path=ddf_path, out_path="")
    assert os.path.isfile(out_path)
    os.remove(out_path)


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
