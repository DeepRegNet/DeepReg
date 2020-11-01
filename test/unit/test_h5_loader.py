"""
Tests functionality of the H5FileLoader
"""
from test.unit.util import is_equal_np
from typing import List

import numpy as np
import pytest

from deepreg.dataset.loader.h5_loader import H5FileLoader


def get_loader_h5_file_names(loader: H5FileLoader) -> List[str]:
    """return the h5 file names in the H5FileLoader"""
    return [f.filename for f in loader.h5_files.values()]


class TestH5FileLoader:
    def test_init_paired(self):
        # test h5_files and data_path_splits
        dir_paths = ["./data/test/h5/paired/test"]
        name = "fixed_images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        got = [get_loader_h5_file_names(loader), loader.data_path_splits]
        expected = [
            ["./data/test/h5/paired/test/fixed_images.h5"],
            [("./data/test/h5/paired/test", "case000025.nii.gz")],
        ]
        loader.close()
        assert got == expected

    def test_init_unpaired(self):
        # test h5_files and data_path_splits
        dir_paths = ["./data/test/h5/unpaired/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        got = [get_loader_h5_file_names(loader), loader.data_path_splits]
        expected = [
            ["./data/test/h5/unpaired/test/images.h5"],
            [
                ("./data/test/h5/unpaired/test", "case000025.nii.gz"),
                ("./data/test/h5/unpaired/test", "case000026.nii.gz"),
            ],
        ]
        loader.close()
        assert got == expected

    def test_init_grouped(self):
        # test h5_files, data_path_splits and group_struct
        dir_paths = ["./data/test/h5/grouped/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=True)
        got = [
            get_loader_h5_file_names(loader),
            loader.data_path_splits,
            loader.group_struct,
        ]
        expected = [
            ["./data/test/h5/grouped/test/images.h5"],
            [
                ("./data/test/h5/grouped/test", "1", "1"),
                ("./data/test/h5/grouped/test", "1", "2"),
            ],
            [[0, 1]],
        ]
        loader.close()
        assert got == expected

    def test_init_multi_dirs_grouped(self):
        # test h5_files, data_path_splits and group_struct
        dir_paths = ["./data/test/h5/grouped/train", "./data/test/h5/grouped/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=True)
        got = [
            get_loader_h5_file_names(loader),
            loader.data_path_splits,
            loader.group_struct,
        ]
        expected = [
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
        ]
        loader.close()
        assert got == expected

    def test_init_duplicated_dirs(self):
        # duplicated dir_paths
        dir_paths = ["./data/test/h5/grouped/test", "./data/test/h5/grouped/test"]
        name = "images"
        with pytest.raises(ValueError) as err_info:
            H5FileLoader(dir_paths=dir_paths, name=name, grouped=True)
        assert "dir_paths have repeated elements" in str(err_info.value)

    @pytest.mark.parametrize(
        "name,err_msg",
        [
            ("images", "does not exist"),  # test not existed files
            (
                "fixed_images",
                "h5_file keys must be of form group-X-Y",
            ),  # test wrong keys for grouped data
        ],
    )
    def test_set_data_structure(self, name, err_msg):
        """
        test set_data_structure in addition to tests above in test_init
        """
        dir_paths = ["./data/test/h5/paired/test"]
        with pytest.raises(AssertionError) as err_info:
            H5FileLoader(dir_paths=dir_paths, name=name, grouped=True)
        assert err_msg in str(err_info.value)

    def test_set_group_structure(self):
        """
        test set_group_structure in addition to tests above in test_init
        """
        # data is not grouped but try using group_struct
        dir_paths = ["./data/test/h5/paired/test"]
        name = "fixed_images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        with pytest.raises(AttributeError) as err_info:
            loader.group_struct
        assert "object has no attribute" in str(err_info.value)

    def test_get_data_paired(self):
        """
        test get_data method by verifying returns' shape and value stats
        """
        dir_paths = ["./data/test/h5/paired/test"]
        name = "fixed_images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        index = 0
        array = loader.get_data(index)
        got = [
            np.shape(array),
            [np.amax(array), np.amin(array), np.mean(array), np.std(array)],
        ]
        expected = [(44, 59, 41), [255.0, 0.0, 68.359276, 65.84009]]
        loader.close()
        assert got[0] == expected[0]
        assert is_equal_np(got[1], expected[1])

    def test_get_data_unpaired(self):
        """
        test get_data method by verifying returns' shape and value stats
        """
        dir_paths = ["./data/test/h5/unpaired/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        index = 0
        array = loader.get_data(index)
        got = [
            np.shape(array),
            [np.amax(array), np.amin(array), np.mean(array), np.std(array)],
        ]
        expected = [(64, 64, 60), [255.0, 0.0, 60.073948, 47.27648]]
        loader.close()
        assert got[0] == expected[0]
        assert is_equal_np(got[1], expected[1])

    def test_get_data_grouped(self):
        """
        test get_data method by verifying returns' shape and value stats
        """
        dir_paths = ["./data/test/h5/grouped/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=True)
        index = (0, 1)
        array = loader.get_data(index)
        got = [
            np.shape(array),
            [np.amax(array), np.amin(array), np.mean(array), np.std(array)],
        ]
        expected = [(64, 64, 60), [255.0, 0.0, 60.073948, 47.27648]]
        loader.close()
        assert got[0] == expected[0]
        assert is_equal_np(got[1], expected[1])

    def test_get_data_multi_dirs_grouped(self):
        """
        test get_data method by verifying returns' shape and value stats
        """
        dir_paths = ["./data/test/h5/grouped/train", "./data/test/h5/grouped/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=True)
        index = (0, 1)
        array = loader.get_data(index)
        got = [
            np.shape(array),
            [np.amax(array), np.amin(array), np.mean(array), np.std(array)],
        ]
        expected = [(64, 64, 60), [255.0, 0.0, 60.073948, 47.27648]]
        loader.close()
        assert got[0] == expected[0]
        assert is_equal_np(got[1], expected[1])

    def test_get_data_ids_paired(self):
        """
        check if the get_data_ids method works as expected
        """
        dir_paths = ["./data/test/h5/paired/test"]
        name = "fixed_images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        got = loader.get_data_ids()
        expected = [("./data/test/h5/paired/test", "case000025.nii.gz")]
        loader.close()
        assert got == expected

    def test_get_data_ids_unpaired(self):
        """
        check if the get_data_ids method works as expected
        """
        dir_paths = ["./data/test/h5/unpaired/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        got = loader.get_data_ids()
        expected = [
            ("./data/test/h5/unpaired/test", "case000025.nii.gz"),
            ("./data/test/h5/unpaired/test", "case000026.nii.gz"),
        ]
        loader.close()
        assert got == expected

    def test_get_data_ids_grouped(self):
        """
        check if the get_data_ids method works as expected
        """
        dir_paths = ["./data/test/h5/grouped/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        got = loader.get_data_ids()
        expected = [
            ("./data/test/h5/grouped/test", "group-1-1"),
            ("./data/test/h5/grouped/test", "group-1-2"),
        ]
        loader.close()
        assert got == expected

    def test_get_data_ids_multi_dirs_grouped(self):
        """
        check if the get_data_ids method works as expected
        """
        dir_paths = ["./data/test/h5/grouped/train", "./data/test/h5/grouped/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        got = loader.get_data_ids()
        expected = [
            ("./data/test/h5/grouped/train", "group-1-1"),
            ("./data/test/h5/grouped/train", "group-1-2"),
            ("./data/test/h5/grouped/train", "group-1-3"),
            ("./data/test/h5/grouped/train", "group-1-4"),
            ("./data/test/h5/grouped/train", "group-2-1"),
            ("./data/test/h5/grouped/train", "group-2-2"),
            ("./data/test/h5/grouped/train", "group-2-3"),
            ("./data/test/h5/grouped/test", "group-1-1"),
            ("./data/test/h5/grouped/test", "group-1-2"),
        ]
        loader.close()
        assert got == expected

        # wrong index for paired
        dir_paths = ["./data/test/h5/paired/test"]
        name = "fixed_images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        with pytest.raises(AssertionError):
            # negative data_index
            loader.get_data(index=-1)
        with pytest.raises(IndexError):
            # out of range data_index
            loader.get_data(index=64)
        with pytest.raises(AssertionError):
            # non int data_index
            loader.get_data(index=(0, 1))
        with pytest.raises(ValueError) as err_info:
            loader.get_data(index="wrong")
        assert "must be int, or tuple" in str(err_info.value)
        loader.close()

        # wrong index for grouped
        dir_paths = ["./data/test/h5/grouped/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=True)
        with pytest.raises(AssertionError):
            # non-tuple data_index
            loader.get_data(index=1)
        loader.close()

    def test_get_num_images_paired(self):
        """
        check if the get_num_images method works as expected
        """
        dir_paths = ["./data/test/h5/paired/test"]
        name = "fixed_images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        got = loader.get_num_images()
        expected = 1
        loader.close()
        assert got == expected

    def test_get_num_images_unpaired(self):
        """
        check if the get_num_images method works as expected
        """
        dir_paths = ["./data/test/h5/unpaired/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        got = loader.get_num_images()
        expected = 2
        loader.close()
        assert got == expected

    def test_get_num_images_grouped(self):
        """
        check if the get_num_images method works as expected
        """
        dir_paths = ["./data/test/h5/grouped/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=True)
        got = loader.get_num_images()
        expected = 2
        loader.close()
        assert got == expected

    def test_get_num_images_multi_dirs_grouped(self):
        """
        check if the get_num_images method works as expected
        """
        dir_paths = ["./data/test/h5/grouped/train", "./data/test/h5/grouped/test"]
        name = "images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=True)
        got = loader.get_num_images()
        expected = 9
        loader.close()
        assert got == expected

    def test_close(self):
        """
        check if close method works as intended and closes file
        close is the same code for all cases, so no need to test all cases
        """
        # paired
        dir_paths = ["./data/test/h5/paired/test"]
        name = "fixed_images"
        loader = H5FileLoader(dir_paths=dir_paths, name=name, grouped=False)
        loader.close()
        for f in loader.h5_files.values():
            assert not f.__bool__()
