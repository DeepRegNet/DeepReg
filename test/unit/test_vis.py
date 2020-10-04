import os

import pytest

from deepreg.vis import main, string_to_list


def test_string_to_list():
    string = "abc, 123, def, 456"
    got = string_to_list(string)
    expected = ["abc", "123", "def", "456"]
    assert got == expected


def test_gif_slices():
    """Covered by test_main"""
    pass


def test_tile_slices():
    """Covered by test_main"""
    pass


def test_gif_warp():
    """Covered by test_main"""
    pass


def test_gif_tile_slices():
    """Covered by test_main"""
    pass


def test_main():
    img_paths = "./data/test/nifti/unit_test/moving_image.nii.gz"
    ddf_path = "./data/test/nifti/unit_test/ddf.nii.gz"

    # test mode 0 and check output
    out_path = "logs/moving_image.gif"
    main(
        args=[
            "--mode",
            0,
            "--image-paths",
            img_paths,
            "--save-path",
            "logs",
            "--interval",
            "50",
        ]
    )
    assert os.path.exists(out_path)
    os.remove(out_path)

    # test mode 1 and check output
    out_path_1 = "logs/moving_image_slice_1.gif"
    out_path_2 = "logs/moving_image_slice_2.gif"
    out_path_3 = "logs/moving_image_slice_3.gif"
    main(
        args=[
            "--mode",
            "1",
            "--image-paths",
            img_paths,
            "--save-path",
            "logs",
            "--interval",
            "50",
            "--num-interval",
            "100",
            "--slice-inds",
            "1,2,3",
            "--ddf-path",
            ddf_path,
        ]
    )
    assert os.path.exists(out_path_1)
    assert os.path.exists(out_path_2)
    assert os.path.exists(out_path_3)
    os.remove(out_path_1)
    os.remove(out_path_2)
    os.remove(out_path_3)

    # test mode 1 and check output when no slice_inds
    out_path_partial = "moving_image_slice"
    main(
        args=[
            "--mode",
            "1",
            "--image-paths",
            img_paths,
            "--save-path",
            "logs",
            "--interval",
            "50",
            "--num-interval",
            "100",
            "--ddf-path",
            ddf_path,
        ]
    )
    assert any([out_path_partial in file for file in os.listdir("logs")])

    # test mode 1 and check if exception is caught
    with pytest.raises(Exception) as err_info:
        main(
            args=[
                "--mode",
                "1",
                "--image-paths",
                img_paths,
                "--save-path",
                "logs",
                "--interval",
                "50",
                "--num-interval",
                "100",
                "--slice-inds",
                "1,2,3",
            ]
        )
    assert "--ddf-path is required when using --mode 1" in str(err_info.value)

    # test mode 2 and check output
    out_path = "logs/visualisation.png"
    main(
        args=[
            "--mode",
            "2",
            "--image-paths",
            img_paths,
            "--save-path",
            "logs",
            "--slice-inds",
            "1,2,3",
            "--fname",
            "visualisation.png",
            "--col-titles",
            "abc",
        ]
    )
    assert os.path.exists(out_path)
    os.remove(out_path)

    # test mode 2 and check output when no slice_inds and no fname
    out_path = "logs/visualisation.png"
    main(
        args=[
            "--mode",
            "2",
            "--image-paths",
            img_paths,
            "--save-path",
            "logs",
            "--col-titles",
            "abc",
        ]
    )
    assert os.path.exists(out_path)
    os.remove(out_path)

    # test mode 3 and check output
    out_path = "logs/visualisation.gif"
    img_paths_moded = (
        img_paths
        + ","
        + img_paths
        + ","
        + img_paths
        + ","
        + img_paths
        + ","
        + img_paths
        + ","
        + img_paths
    )
    main(
        args=[
            "--mode",
            "3",
            "--image-paths",
            img_paths_moded,
            "--save-path",
            "logs",
            "--interval",
            "50",
            "--size",
            "2,3",
            "--fname",
            "visualisation.gif",
        ]
    )
    assert os.path.exists(out_path)
    os.remove(out_path)

    # test mode 3 and check output when fname not provided
    out_path = "logs/visualisation.gif"
    img_paths_moded = (
        img_paths
        + ","
        + img_paths
        + ","
        + img_paths
        + ","
        + img_paths
        + ","
        + img_paths
        + ","
        + img_paths
    )
    main(
        args=[
            "--mode",
            "3",
            "--image-paths",
            img_paths_moded,
            "--save-path",
            "logs",
            "--interval",
            "50",
            "--size",
            "2,3",
        ]
    )
    assert os.path.exists(out_path)
    os.remove(out_path)

    # test mode 3 and check if exception is caught
    img_paths_moded = (
        img_paths
        + ","
        + img_paths
        + ","
        + img_paths
        + ","
        + img_paths
        + ","
        + img_paths
        + ","
        + img_paths
    )
    with pytest.raises(Exception) as err_info:
        main(
            args=[
                "--mode",
                "3",
                "--image-paths",
                img_paths_moded,
                "--save-path",
                "logs",
                "--interval",
                "50",
                "--size",
                "2,1",
                "--fname",
                "visualisation.gif",
            ]
        )
    assert "The number of images supplied is " in str(err_info.value)
