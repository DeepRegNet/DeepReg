"""
Tests functionality of the H5FileLoader
"""
import os
from test.unit.util import is_equal_np
from typing import List

import h5py
import numpy as np
import pytest

from deepreg.dataset.loader.h5_loader import H5FileLoader


def get_loader_h5_file_names(loader: H5FileLoader) -> List[str]:
    return [f.filename for f in loader.h5_files.values()]  # type: ignore


def get_loader(loader_name):
    if loader_name in [
        "paired",
        "unpaired",
        "grouped",
    ]:
        dir_paths = [f"./data/test/h5/{loader_name}/test"]
        name = "fixed_images" if loader_name == "paired" else "images"
        grouped = loader_name == "grouped"
    elif loader_name == "multi_dirs_grouped":
        dir_paths = ["./data/test/h5/grouped/train", "./data/test/h5/grouped/test"]
        name = "images"
        grouped = True
    else:
        raise ValueError
    loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=grouped)
    return loader


class TestH5FileLoader:
    @pytest.mark.parametrize(
        "name,expected",
        [
            (
                "paired",
                [
                    ["./data/test/h5/paired/test/fixed_images.h5"],
                    [("./data/test/h5/paired/test", "case000025.nii.gz")],
                    None,
                ],
            ),
            (
                "unpaired",
                [
                    ["./data/test/h5/unpaired/test/images.h5"],
                    [
                        ("./data/test/h5/unpaired/test", "case000025.nii.gz"),
                        ("./data/test/h5/unpaired/test", "case000026.nii.gz"),
                    ],
                    None,
                ],
            ),
            (
                "grouped",
                [
                    ["./data/test/h5/grouped/test/images.h5"],
                    [
                        ("./data/test/h5/grouped/test", "1", "1"),
                        ("./data/test/h5/grouped/test", "1", "2"),
                    ],
                    [[0, 1]],
                ],
            ),
            (
                "multi_dirs_grouped",
                [
                    [
                        "./data/test/h5/grouped/train/images.h5",
                        "./data/test/h5/grouped/test/images.h5",
                    ],
                    [
                        ("./data/test/h5/grouped/train", "1", "1"),
                        ("./data/test/h5/grouped/train", "1", "2"),
                        ("./data/test/h5/grouped/train", "1", "3"),
                        ("./data/test/h5/grouped/train", "1", "4"),
                        ("./data/test/h5/grouped/train", "2", "1"),
                        ("./data/test/h5/grouped/train", "2", "2"),
                        ("./data/test/h5/grouped/train", "2", "3"),
                        ("./data/test/h5/grouped/test", "1", "1"),
                        ("./data/test/h5/grouped/test", "1", "2"),
                    ],
                    [[7, 8], [0, 1, 2, 3], [4, 5, 6]],
                ],
            ),
        ],
    )
    def test_init(self, name, expected):
        loader = get_loader(name)
        got = [
            get_loader_h5_file_names(loader),
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
            H5FileLoader(dir_paths=dir_paths, name=loader.name, grouped=loader.grouped)
        assert "dir_paths have repeated elements" in str(err_info.value)
        # no need to close files as they haven't been opened yet

    @pytest.mark.parametrize(
        "name,err_msg",
        [
            ("images.h5 does not exist", "does not exist"),  # test not existed files
            (
                "fixed_images",
                "h5_file keys must be of form group-X-Y",
            ),  # test wrong keys for grouped data
        ],
    )
    def test_set_data_structure_err1(self, name, err_msg):
        dir_paths = ["./data/test/h5/paired/test"]
        with pytest.raises(AssertionError) as err_info:
            H5FileLoader(dir_paths=dir_paths, name=name, grouped=True)
        assert err_msg in str(err_info.value)

    def test_set_data_structure_err2(self):
        dir_paths = ["./data/test/h5/paired/test"]
        name = "error"
        file_path = os.path.join(dir_paths[0], f"{name}.h5")
        with h5py.File(file_path, "w"):
            pass
        with pytest.raises(ValueError) as err_info:
            H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        assert "No data collected" in str(err_info.value)
        os.remove(file_path)

    @pytest.mark.parametrize(
        "name,index,expected",
        [
            ("paired", 0, [(44, 59, 41), [255.0, 0.0, 68.359276, 65.84009]]),
            ("unpaired", 0, [(64, 64, 60), [255.0, 0.0, 60.073948, 47.27648]]),
            ("grouped", (0, 1), [(64, 64, 60), [255.0, 0.0, 60.073948, 47.27648]]),
            (
                "multi_dirs_grouped",
                (0, 1),
                [(64, 64, 60), [255.0, 0.0, 60.073948, 47.27648]],
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
            ("paired", [("./data/test/h5/paired/test", "case000025.nii.gz")]),
            (
                "unpaired",
                [
                    ("./data/test/h5/unpaired/test", "case000025.nii.gz"),
                    ("./data/test/h5/unpaired/test", "case000026.nii.gz"),
                ],
            ),
            (
                "grouped",
                [
                    ("./data/test/h5/grouped/test", "1", "1"),
                    ("./data/test/h5/grouped/test", "1", "2"),
                ],
            ),
            (
                "multi_dirs_grouped",
                [
                    ("./data/test/h5/grouped/train", "1", "1"),
                    ("./data/test/h5/grouped/train", "1", "2"),
                    ("./data/test/h5/grouped/train", "1", "3"),
                    ("./data/test/h5/grouped/train", "1", "4"),
                    ("./data/test/h5/grouped/train", "2", "1"),
                    ("./data/test/h5/grouped/train", "2", "2"),
                    ("./data/test/h5/grouped/train", "2", "3"),
                    ("./data/test/h5/grouped/test", "1", "1"),
                    ("./data/test/h5/grouped/test", "1", "2"),
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
        [("paired", 1), ("unpaired", 2), ("grouped", 2), ("multi_dirs_grouped", 9)],
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
        for f in loader.h5_files.values():
            assert not f.__bool__()
