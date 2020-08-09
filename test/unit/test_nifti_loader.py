"""
Tests functionality of the NiftiFileLoader
"""
from test.unit.util import is_equal_np

import numpy as np
import pytest

from deepreg.dataset.loader.nifti_loader import NiftiFileLoader, load_nifti_file


def test_load_nifti_file():
    """
    check if the nifti files can be correctly loaded
    """

    # nii.gz
    nii_gz_filepath = "./data/test/nifti/paired/test/fixed_images/case000026.nii.gz"
    load_nifti_file(file_path=nii_gz_filepath)

    # nii
    nii_filepath = "./data/test/nifti/unit_test/case000026.nii"
    load_nifti_file(file_path=nii_filepath)

    # wrong file type
    h5_filepath = "./data/test/h5/paired/test/fixed_images.h5"
    with pytest.raises(ValueError) as err_info:
        load_nifti_file(file_path=h5_filepath)
    assert "Nifti file path must end with .nii or .nii.gz" in str(err_info.value)


def test_init():
    """
    check init of NiftiFileLoader
    this includes some test on set_data_structure/set_group_structure
    as these functions are called in init
    """

    # paired
    # test data_path_splits
    dir_paths = ["./data/test/nifti/paired/test"]
    name = "fixed_images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
    got = loader.data_path_splits
    expected = [
        ("./data/test/nifti/paired/test", "case000025", "nii.gz"),
        ("./data/test/nifti/paired/test", "case000026", "nii.gz"),
    ]
    loader.close()
    assert got == expected

    # unpaired
    # test data_path_splits
    dir_paths = ["./data/test/nifti/unpaired/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
    got = loader.data_path_splits
    expected = [
        ("./data/test/nifti/unpaired/test", "case000025", "nii.gz"),
        ("./data/test/nifti/unpaired/test", "case000026", "nii"),
    ]
    loader.close()
    assert got == expected

    # grouped
    # test data_path_splits and group_struct
    dir_paths = ["./data/test/nifti/grouped/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    got = [loader.data_path_splits, loader.group_struct]
    expected = [
        [
            ("./data/test/nifti/grouped/test", "group1", "case000025", "nii.gz"),
            ("./data/test/nifti/grouped/test", "group1", "case000026", "nii.gz"),
        ],
        [[0, 1]],
    ]
    loader.close()
    assert got == expected

    # multi dirs
    # test data_path_splits and group_struct
    dir_paths = ["./data/test/nifti/grouped/train", "./data/test/nifti/grouped/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    got = [loader.data_path_splits, loader.group_struct]
    expected = [
        [
            ("./data/test/nifti/grouped/test", "group1", "case000025", "nii.gz"),
            ("./data/test/nifti/grouped/test", "group1", "case000026", "nii.gz"),
            ("./data/test/nifti/grouped/train", "group1", "case000000", "nii.gz"),
            ("./data/test/nifti/grouped/train", "group1", "case000001", "nii.gz"),
            ("./data/test/nifti/grouped/train", "group1", "case000003", "nii.gz"),
            ("./data/test/nifti/grouped/train", "group1", "case000008", "nii.gz"),
            ("./data/test/nifti/grouped/train", "group2", "case000009", "nii.gz"),
            ("./data/test/nifti/grouped/train", "group2", "case000011", "nii.gz"),
            ("./data/test/nifti/grouped/train", "group2", "case000012", "nii.gz"),
        ],
        [[0, 1], [2, 3, 4, 5], [6, 7, 8]],
    ]
    loader.close()
    assert got == expected

    # duplicated dir_paths
    dir_paths = ["./data/test/nifti/grouped/test", "./data/test/nifti/grouped/test"]
    name = "images"
    with pytest.raises(ValueError) as err_info:
        NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    assert "dir_paths have repeated elements" in str(err_info.value)


def test_set_data_structure():
    """
    test set_data_structure in addition to tests above in test_init
    """
    # test not existed directories
    dir_paths = ["./data/test/nifti/paired/test"]
    name = "images"
    with pytest.raises(AssertionError) as err_info:
        NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    assert "does not exist" in str(err_info.value)


def test_set_group_structure():
    """
    test set_group_structure in addition to tests above in test_init
    """
    # data is not grouped but try using group_struct
    dir_paths = ["./data/test/nifti/paired/test"]
    name = "fixed_images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
    with pytest.raises(AttributeError) as err_info:
        loader.group_struct
    assert "object has no attribute" in str(err_info.value)


def test_get_data():
    """
    test get_data method by verifying returns' shape and value stats
    """
    # paired
    dir_paths = ["./data/test/nifti/paired/test"]
    name = "fixed_images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
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

    # unpaired
    dir_paths = ["./data/test/nifti/unpaired/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
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

    # grouped
    dir_paths = ["./data/test/nifti/grouped/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    index = (0, 1)
    array = loader.get_data(index)
    got = [
        np.shape(array),
        [np.amax(array), np.amin(array), np.mean(array), np.std(array)],
    ]
    expected = [(64, 64, 60), [255.0, 0.0, 85.67942, 49.193127]]
    loader.close()
    assert got[0] == expected[0]
    assert is_equal_np(got[1], expected[1])

    # multi dirs
    dir_paths = ["./data/test/nifti/grouped/train", "./data/test/nifti/grouped/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    index = (0, 1)
    array = loader.get_data(index)
    got = [
        np.shape(array),
        [np.amax(array), np.amin(array), np.mean(array), np.std(array)],
    ]
    expected = [(64, 64, 60), [255.0, 0.0, 85.67942, 49.193127]]
    loader.close()
    assert got[0] == expected[0]
    assert is_equal_np(got[1], expected[1])


def test_get_data_ids():
    """
    check if the get_data_ids method works as expected
    """
    # paired
    dir_paths = ["./data/test/nifti/paired/test"]
    name = "fixed_images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
    got = loader.get_data_ids()
    expected = [
        ("./data/test/nifti/paired/test", "case000025"),
        ("./data/test/nifti/paired/test", "case000026"),
    ]
    loader.close()
    assert got == expected

    # unpaired
    dir_paths = ["./data/test/nifti/unpaired/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
    got = loader.get_data_ids()
    expected = [
        ("./data/test/nifti/unpaired/test", "case000025"),
        ("./data/test/nifti/unpaired/test", "case000026"),
    ]
    loader.close()
    assert got == expected

    # grouped
    dir_paths = ["./data/test/nifti/grouped/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    got = loader.get_data_ids()
    expected = [
        ("./data/test/nifti/grouped/test", "group1", "case000025"),
        ("./data/test/nifti/grouped/test", "group1", "case000026"),
    ]
    loader.close()
    assert got == expected

    # multi dirs
    dir_paths = ["./data/test/nifti/grouped/train", "./data/test/nifti/grouped/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    got = loader.get_data_ids()
    expected = [
        ("./data/test/nifti/grouped/test", "group1", "case000025"),
        ("./data/test/nifti/grouped/test", "group1", "case000026"),
        ("./data/test/nifti/grouped/train", "group1", "case000000"),
        ("./data/test/nifti/grouped/train", "group1", "case000001"),
        ("./data/test/nifti/grouped/train", "group1", "case000003"),
        ("./data/test/nifti/grouped/train", "group1", "case000008"),
        ("./data/test/nifti/grouped/train", "group2", "case000009"),
        ("./data/test/nifti/grouped/train", "group2", "case000011"),
        ("./data/test/nifti/grouped/train", "group2", "case000012"),
    ]
    loader.close()
    assert got == expected

    # wrong index for paired
    dir_paths = ["./data/test/nifti/paired/test"]
    name = "fixed_images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
    with pytest.raises(AssertionError):
        loader.get_data(index=(0, 1))
    with pytest.raises(ValueError) as err_info:
        loader.get_data(index=[0])
    assert "must be int, or tuple" in str(err_info.value)

    # wrong index for grouped
    dir_paths = ["./data/test/nifti/grouped/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    with pytest.raises(AssertionError):
        # negative group_index
        loader.get_data(index=(-1, 1))
    with pytest.raises(IndexError):
        # out of range group_index
        loader.get_data(index=(32, 1))
    with pytest.raises(AssertionError):
        # negative in_group_data_index
        loader.get_data(index=(0, -1))
    with pytest.raises(IndexError):
        # out of range in_group_data_index
        loader.get_data(index=(0, 32))


def test_get_num_images():
    """
    check if the get_num_images method works as expected
    """
    # paired
    dir_paths = ["./data/test/nifti/paired/test"]
    name = "fixed_images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
    got = loader.get_num_images()
    expected = 2
    loader.close()
    assert got == expected

    # unpaired
    dir_paths = ["./data/test/nifti/unpaired/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
    got = loader.get_num_images()
    expected = 2
    loader.close()
    assert got == expected

    # grouped
    dir_paths = ["./data/test/nifti/grouped/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    got = loader.get_num_images()
    expected = 2
    loader.close()
    assert got == expected

    # multi dirs
    dir_paths = ["./data/test/nifti/grouped/train", "./data/test/nifti/grouped/test"]
    name = "images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=True)
    got = loader.get_num_images()
    expected = 9
    loader.close()
    assert got == expected


def test_close():
    """
    check if close method works as intended
    close is the same code for all cases, so no need to test all cases
    """
    # paired
    dir_paths = ["./data/test/nifti/paired/test"]
    name = "fixed_images"
    loader = NiftiFileLoader(dir_paths=dir_paths, name=name, grouped=False)
    loader.close()
