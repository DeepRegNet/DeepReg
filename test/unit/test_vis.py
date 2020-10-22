import os

import pytest

from deepreg.vis import main, string_to_list


def test_string_to_list():
    string = "abc, 123, def, 456"
    got = string_to_list(string)
    expected = ["abc", "123", "def", "456"]
    assert got == expected


class TestMain:
    img_paths = "./data/test/nifti/unit_test/moving_image.nii.gz"
    ddf_path = "./data/test/nifti/unit_test/ddf.nii.gz"

    def test_mode0(self):
        # test mode 0 and check output
        out_path = "logs/moving_image.gif"
        main(
            args=[
                "--mode",
                0,
                "--image-paths",
                self.img_paths,
                "--save-path",
                "logs",
                "--interval",
                "50",
            ]
        )
        assert os.path.exists(out_path)
        os.remove(out_path)

    def test_mode1_output(self):
        # test mode 1 and check output
        out_path_1 = "logs/moving_image_slice_1.gif"
        out_path_2 = "logs/moving_image_slice_2.gif"
        out_path_3 = "logs/moving_image_slice_3.gif"
        main(
            args=[
                "--mode",
                "1",
                "--image-paths",
                self.img_paths,
                "--save-path",
                "logs",
                "--interval",
                "50",
                "--num-interval",
                "100",
                "--slice-inds",
                "1,2,3",
                "--ddf-path",
                self.ddf_path,
            ]
        )
        assert os.path.exists(out_path_1)
        assert os.path.exists(out_path_2)
        assert os.path.exists(out_path_3)
        os.remove(out_path_1)
        os.remove(out_path_2)
        os.remove(out_path_3)

    def test_mode1_no_slidce_inds(self):
        # test mode 1 and check output when no slice_inds
        out_path_partial = "moving_image_slice"
        main(
            args=[
                "--mode",
                "1",
                "--image-paths",
                self.img_paths,
                "--save-path",
                "logs",
                "--interval",
                "50",
                "--num-interval",
                "100",
                "--ddf-path",
                self.ddf_path,
            ]
        )
        assert any([out_path_partial in file for file in os.listdir("logs")])

    def test_mode1_err(self):
        # test mode 1 and check if exception is caught
        with pytest.raises(Exception) as err_info:
            main(
                args=[
                    "--mode",
                    "1",
                    "--image-paths",
                    self.img_paths,
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

    @pytest.mark.parametrize(
        "extra_args",
        [
            [
                "--slice-inds",
                "1,2,3",
                "--fname",
                "visualisation.png",
                "--col-titles",
                "abc",
            ],
            [
                "--slice-inds",
                "1,2,3",
                "--fname",
                "visualisation.png",
            ],  # no col_titles
            [
                "--col-titles",
                "abc",
            ],  # no slice_inds and no fname
        ],
    )
    def test_mode2_output(self, extra_args):
        # test mode 2 and check output
        common_args = [
            "--mode",
            "2",
            "--image-paths",
            self.img_paths,
            "--save-path",
            "logs",
        ]
        out_path = "logs/visualisation.png"
        main(args=common_args + extra_args)
        assert os.path.exists(out_path)
        os.remove(out_path)

    @pytest.mark.parametrize(
        "extra_args",
        [
            [
                "--fname",
                "visualisation.gif",
            ],
            [],  # no fname
        ],
    )
    def test_case3(self, extra_args):
        # test mode 3 and check output
        out_path = "logs/visualisation.gif"
        img_paths_moded = ",".join([self.img_paths] * 6)
        common_args = [
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
        main(
            args=common_args + extra_args,
        )
        assert os.path.exists(out_path)
        os.remove(out_path)

    def test_mode3_num_image_err(self):
        # test mode 3 and check if exception is caught
        img_paths = ",".join([self.img_paths] * 6)
        with pytest.raises(Exception) as err_info:
            main(
                args=[
                    "--mode",
                    "3",
                    "--image-paths",
                    img_paths,
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

    def test_mode3_shape_err(self):
        # test mode 3 and check if exception is caught
        new_img_path = "data/test/nifti/paired/test/fixed_images/case000025.nii.gz"
        img_paths_moded = self.img_paths + "," + new_img_path
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
        assert "all images do not have equal shapes" in str(err_info.value)
