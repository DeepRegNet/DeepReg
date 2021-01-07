"""
Tests functionality of the NiftiFileLoader
"""
import os
from test.unit.util import is_equal_np

import numpy as np
import pytest

from deepreg.dataset.loader.nifti_loader import NiftiFileLoader, load_nifti_file


def get_loader(loader_name):
    if loader_name in [
        "paired",
        "unpaired",
        "grouped",
    ]:
        dir_paths = [f"./data/test/nifti/{loader_name}/test"]
        name = "fixed_images" if loader_name == "paired" else "images"
        grouped = loader_name == "grouped"
    elif loader_name == "multi_dirs_grouped":
        dir_paths = [
            "./data/test/nifti/grouped/train",
            "./data/test/nifti/grouped/test",
        ]
        name = "images"
        grouped = True
    else:
        raise ValueError
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=grouped)
    return loader


@pytest.mark.parametrize(
    "path,shape",
    [
        ("./data/test/nifti/paired/test/fixed_images/case000026.nii.gz", (44, 59, 41)),
        ("./data/test/nifti/unit_test/case000026.nii", (44, 59, 41)),
    ],
)
def test_load_nifti_file(path, shape):
    arr = load_nifti_file(file_path=path)
    assert arr.shape == shape


def test_load_nifti_file_err():
    h5_filepath = "./data/test/h5/paired/test/fixed_images.h5"
    with pytest.raises(ValueError) as err_info:
        load_nifti_file(file_path=h5_filepath)
    assert "Nifti file path must end with .nii or .nii.gz" in str(err_info.value)


class TestNiftiFileLoader:
    @pytest.mark.parametrize(
        "name,expected",
        [
            (
                "paired",
                [
                    [
                        ("./data/test/nifti/paired/test", "case000025", "nii.gz"),
                        ("./data/test/nifti/paired/test", "case000026", "nii.gz"),
                    ],
                    None,
                ],
            ),
            (
                "unpaired",
                [
                    [
                        ("./data/test/nifti/unpaired/test", "case000025", "nii.gz"),
                        ("./data/test/nifti/unpaired/test", "case000026", "nii"),
                    ],
                    None,
                ],
            ),
            (
                "grouped",
                [
                    [
                        (
                            "./data/test/nifti/grouped/test",
                            "group1",
                            "case000025",
                            "nii.gz",
                        ),
                        (
                            "./data/test/nifti/grouped/test",
                            "group1",
                            "case000026",
                            "nii.gz",
                        ),
                    ],
                    [[0, 1]],
                ],
            ),
            (
                "multi_dirs_grouped",
                [
                    [
                        (
                            "./data/test/nifti/grouped/test",
                            "group1",
                            "case000025",
                            "nii.gz",
                        ),
                        (
                            "./data/test/nifti/grouped/test",
                            "group1",
                            "case000026",
                            "nii.gz",
                        ),
                        (
                            "./data/test/nifti/grouped/train",
                            "group1",
                            "case000000",
                            "nii.gz",
                        ),
                        (
                            "./data/test/nifti/grouped/train",
                            "group1",
                            "case000001",
                            "nii.gz",
                        ),
                        (
                            "./data/test/nifti/grouped/train",
                            "group1",
                            "case000003",
                            "nii.gz",
                        ),
                        (
                            "./data/test/nifti/grouped/train",
                            "group1",
                            "case000008",
                            "nii.gz",
                        ),
                        (
                            "./data/test/nifti/grouped/train",
                            "group2",
                            "case000009",
                            "nii.gz",
                        ),
                        (
                            "./data/test/nifti/grouped/train",
                            "group2",
                            "case000011",
                            "nii.gz",
                        ),
                        (
                            "./data/test/nifti/grouped/train",
                            "group2",
                            "case000012",
                            "nii.gz",
                        ),
                    ],
                    [[0, 1], [2, 3, 4, 5], [6, 7, 8]],
                ],
            ),
        ],
    )
    def test_init(self, name, expected):
        loader = get_loader(name)
        got = [
            loader.data_path_splits,
            loader.group_struct,
        ]
        assert got == expected
        loader.close()

    @pytest.mark.parametrize(
        "name",
        [
            "paired",
            "unpaired",
            "grouped",
        ],
    )
    def test_init_duplicated_dirs(self, name):
        # duplicated dir_paths
        loader = get_loader(name)
        dir_paths = loader.dir_paths * 2
        with pytest.raises(ValueError) as err_info:
            NiftiFileLoader(
                dir_paths=dir_paths, name=loader.name, grouped=loader.grouped
            )
        assert "dir_paths have repeated elements" in str(err_info.value)
        loader.close()

    @pytest.mark.parametrize(
        "name,err_msg",
        [
            (
                "images",
                "directory ./data/test/h5/paired/test/images does not exist",
            ),  # test not existed files
        ],
    )
    def test_set_data_structure_err1(self, name, err_msg):
        dir_paths = ["./data/test/h5/paired/test"]
        with pytest.raises(AssertionError) as err_info:
            NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
        assert err_msg in str(err_info.value)

    def test_set_data_structure_err2(self):
        dir_paths = ["./data/test/nifti/paired/test"]
        name = "error"
        dir_path = os.path.join(dir_paths[0], name)
        os.makedirs(dir_path, exist_ok=True)
        with pytest.raises(ValueError) as err_info:
            NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
        assert "No data collected" in str(err_info.value)
        os.removedirs(dir_path)

    @pytest.mark.parametrize(
        "name,index,expected",
        [
            ("paired", 0, [(44, 59, 41), [255.0, 0.0, 68.359276, 65.84009]]),
            ("unpaired", 0, [(64, 64, 60), [255.0, 0.0, 60.073948, 47.27648]]),
            ("grouped", (0, 1), [(64, 64, 60), [255.0, 0.0, 85.67942, 49.193127]]),
            (
                "multi_dirs_grouped",
                (0, 1),
                [(64, 64, 60), [255.0, 0.0, 85.67942, 49.193127]],
            ),
        ],
    )
    def test_get_data(self, name, index, expected):
        loader = get_loader(name)
        array = loader.get_data(index)
        got = [
            np.shape(array),
            [np.amax(array), np.amin(array), np.mean(array), np.std(array)],
        ]
        assert got[0] == expected[0]
        assert is_equal_np(got[1], expected[1])
        loader.close()

    @pytest.mark.parametrize(
        "name,expected",
        [
            (
                "paired",
                [
                    ("./data/test/nifti/paired/test", "case000025"),
                    ("./data/test/nifti/paired/test", "case000026"),
                ],
            ),
            (
                "unpaired",
                [
                    ("./data/test/nifti/unpaired/test", "case000025"),
                    ("./data/test/nifti/unpaired/test", "case000026"),
                ],
            ),
            (
                "grouped",
                [
                    ("./data/test/nifti/grouped/test", "group1", "case000025"),
                    ("./data/test/nifti/grouped/test", "group1", "case000026"),
                ],
            ),
            (
                "multi_dirs_grouped",
                [
                    ("./data/test/nifti/grouped/test", "group1", "case000025"),
                    ("./data/test/nifti/grouped/test", "group1", "case000026"),
                    ("./data/test/nifti/grouped/train", "group1", "case000000"),
                    ("./data/test/nifti/grouped/train", "group1", "case000001"),
                    ("./data/test/nifti/grouped/train", "group1", "case000003"),
                    ("./data/test/nifti/grouped/train", "group1", "case000008"),
                    ("./data/test/nifti/grouped/train", "group2", "case000009"),
                    ("./data/test/nifti/grouped/train", "group2", "case000011"),
                    ("./data/test/nifti/grouped/train", "group2", "case000012"),
                ],
            ),
        ],
    )
    def test_get_data_ids(self, name, expected):
        loader = get_loader(name)
        got = loader.get_data_ids()
        assert got == expected
        loader.close()

    @pytest.mark.parametrize(
        "index,err_type",
        [
            (-1, AssertionError),
            (64, IndexError),
            ((0, 1), AssertionError),
            ("wrong", ValueError),
        ],
    )
    def test_get_data_ids_check_err_with_paired(self, index, err_type):
        # wrong index for paired
        loader = get_loader("paired")
        with pytest.raises(err_type):
            loader.get_data(index=index)
        loader.close()

    def test_get_data_ids_check_err_with_grouped(self):
        # wrong index for paired
        loader = get_loader("grouped")
        with pytest.raises(AssertionError):
            # non-tuple data_index
            loader.get_data(index=1)
        loader.close()

    @pytest.mark.parametrize(
        "name,expected",
        [("paired", 2), ("unpaired", 2), ("grouped", 2), ("multi_dirs_grouped", 9)],
    )
    def test_get_num_images(self, name, expected):
        loader = get_loader(name)
        got = loader.get_num_images()
        assert got == expected
        loader.close()

    @pytest.mark.parametrize(
        "name",
        [
            "paired",
            "unpaired",
            "grouped",
        ],
    )
    def test_close(self, name):
        loader = get_loader(name)
        loader.close()
