"""
Tests functionality of the NiftiFileLoader
"""
import numpy as np
import pytest

from deepreg.dataset.loader.nifti_loader import NiftiFileLoader, load_nifti_file


def test_load_nifti_file():
    """
    check if the nifti files can be correctly loaded
    """

    # nii.gz
    nii_gz_filepath = "./data/test/nifti/paired/test/fixed_images/case000026.nii.gz"
    load_nifti_file(filepath=nii_gz_filepath)

    # nii
    nii_filepath = "./data/test/nifti/unit_test/case000026.nii"
    load_nifti_file(filepath=nii_filepath)

    # wrong file type
    h5_filepath = "./data/test/h5/paired/test/fixed_images.h5"
    with pytest.raises(ValueError) as err_info:
        load_nifti_file(filepath=h5_filepath)
    assert "Nifti file path must end with .nii or .nii.gz" in str(err_info.value)


def test_init_sufficient_args():
    """
    check if init method of loader returns any errors when all required
    arguments given
    """
    dir_path = "./data/test/nifti/paired/test"
    name = "fixed_images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=False)
    loader.__init__(dir_path=dir_path, name=name, grouped=False)
    loader.close()


def test_file_paths():
    """
    check if the filepaths are the same as expected
    """
    dir_path = "./data/test/nifti/paired/test"
    name = "fixed_images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=False)
    got = loader.file_paths
    expected = [
        "./data/test/nifti/paired/test/fixed_images/case000025.nii.gz",
        "./data/test/nifti/paired/test/fixed_images/case000026.nii.gz",
    ]
    loader.close()
    assert got == expected


def test_set_group_structure():
    """
    check if the set_group_structure method works as intended when data is
    grouped
    """
    dir_path = "./data/test/nifti/grouped/test"
    name = "images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=True)
    loader.set_group_structure()
    got = [loader.group_ids, loader.group_sample_dict]
    expected = [
        ["./data/test/nifti/grouped/test/images/group1"],
        {
            "./data/test/nifti/grouped/test/images/group1": [
                "./data/test/nifti/grouped/test/images/group1/case000025.nii.gz",
                "./data/test/nifti/grouped/test/images/group1/case000026.nii.gz",
            ]
        },
    ]

    loader.close()
    assert got == expected


def test_set_group_structure_ungrouped():
    """
    check if the set_group_structure method works as intended when data is
    not grouped
    """
    dir_path = "./data/test/nifti/paired/test"
    name = "fixed_images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=False)
    loader.set_group_structure()
    with pytest.raises(AttributeError) as err_info:
        loader.group_ids
    assert "object has no attribute" in str(err_info.value)


def test_get_data_ids():
    """
    check if the get_data_ids method works as expected
    """
    dir_path = "./data/test/nifti/paired/test"
    name = "fixed_images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=False)
    got = loader.get_data_ids()
    expected = ["/case000025", "/case000026"]
    loader.close()
    assert got == expected


def test_get_num_images():
    """
    check if the get_num_images method works as expected
    """
    dir_path = "./data/test/nifti/paired/test"
    name = "fixed_images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=False)
    got = int(loader.get_num_images())
    expected = int(2)
    loader.close()
    assert got == expected


def test_close():
    """
    check if close method works as intended
    """
    dir_path = "./data/test/nifti/paired/test"
    name = "fixed_images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=False)
    loader.close()


def check_equal(array1, array2):
    """
    cehck if two arrays are equal
    """
    return np.abs(np.subtract(array1, array2)) < 1e-3


def test_get_data():
    """
    check if the get_data method works as expected and returns array
    as expected
    """
    dir_path = "./data/test/nifti/paired/test"
    name = "fixed_images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=False)
    index = 0
    array = loader.get_data(index)
    got = [
        np.shape(array),
        [np.amax(array), np.amin(array), np.mean(array), np.std(array)],
    ]
    expected = [(44, 59, 41), [255.0, 0.0, 68.359276, 65.84009]]
    loader.close()
    if got[0] == expected[0]:
        assert check_equal(np.array(got[1]), np.array(expected[1])).all()
    else:
        raise AssertionError


def test_get_data_grouped():
    """
    check if the get_data method works as expected and returns array
    as expected
    """
    dir_path = "./data/test/nifti/grouped/test"
    name = "images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=True)
    index = (0, 1)
    array = loader.get_data(index)
    got = [
        np.shape(array),
        [np.amax(array), np.amin(array), np.mean(array), np.std(array)],
    ]
    expected = [(64, 64, 60), [255.0, 0.0, 85.67942, 49.193127]]
    loader.close()
    if got[0] == expected[0]:
        assert check_equal(np.array(got[1]), np.array(expected[1])).all()
    else:
        raise AssertionError


def test_get_data_incorrect_group_index():
    """
    check if the get_data method works as expected and raises error when
    incorrect group index is supplied
    """
    dir_path = "./data/test/nifti/grouped/test"
    name = "images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=True)
    index = (-1, 1)
    with pytest.raises(AssertionError):
        loader.get_data(index)


def test_get_data_negative_sample_index():
    """
    check if the get_data method works as expected and raises error when
    incorrect group index is supplied
    """
    dir_path = "./data/test/nifti/grouped/test"
    name = "images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=True)
    index = (0, -1)
    with pytest.raises(AssertionError):
        loader.get_data(index)


def test_get_data_out_of_range_sample_index():
    """
    check if the get_data method works as expected and raises error when
    incorrect group index is supplied
    """
    dir_path = "./data/test/nifti/grouped/test"
    name = "images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=True)
    index = (0, 32)
    with pytest.raises(AssertionError):
        loader.get_data(index)


def test_get_data_incompatible_args():
    """
    check if the get_data method works as expected and raises an error when
    data is ungrouped but index is not an int
    """
    dir_path = "./data/test/nifti/paired/test"
    name = "fixed_images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=False)
    index = (0, 1)
    with pytest.raises(AssertionError):
        loader.get_data(index)


def test_get_data_incorrect_args():
    """
    check if the get_data method works as expected and raises an error when
    an incorrect data type is fed in
    """
    dir_path = "./data/test/nifti/paired/test"
    name = "fixed_images"

    loader = NiftiFileLoader(dir_path=dir_path, name=name, grouped=False)
    index = "abc"
    with pytest.raises(ValueError) as err_info:
        loader.get_data(index)
    assert "must be int, or tuple" in str(err_info.value)
