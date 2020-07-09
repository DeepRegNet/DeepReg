"""
Tests functionality of the H5FileLoader
"""
import numpy as np
import pytest

from deepreg.dataset.loader.h5_loader import H5FileLoader


def test_init_sufficient_args():
    """
    check if init method of loader returns any errors when all required
    arguments given
    """
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
    loader.__init__(dir_path=dir_path, name=name, grouped=False)
    loader.close()


def test_data_keys():
    """
    check if the data_keys are the same as expected
    """
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
    got = loader.data_keys
    expected = ["case000025.nii.gz"]
    loader.close()
    assert got == expected


def test_h5_file():
    """
    check if the filename is the same as expected
    """
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
    got = loader.h5_file.filename
    expected = "./data/test/h5/paired/test/fixed_images.h5"
    loader.close()
    assert got == expected


def test_set_group_structure():
    """
    check if the set_group_structure method works as intended when data is
    grouped
    """
    dir_path = "./data/test/h5/grouped/test"
    name = "images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=True)
    loader.set_group_structure()
    got = [loader.group_ids, loader.group_sample_dict]
    expected = [["1"], {"1": ["1", "2"]}]
    loader.close()
    assert got == expected


def test_set_group_structure_ungrouped():
    """
    check if the set_group_structure method works as intended when data is
    not grouped
    """
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
    loader.set_group_structure()
    with pytest.raises(AttributeError) as execinfo:
        loader.group_ids
    msg = " ".join(execinfo.value.args[0].split())
    assert "object has no attribute" in msg


def test_get_data_ids():
    """
    check if the get_data_ids method works as expected
    """
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
    got = loader.get_data_ids()
    expected = ["case000025.nii.gz"]
    loader.close()
    assert got == expected


def test_get_num_images():
    """
    check if the get_num_images method works as expected
    """
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
    got = int(loader.get_num_images())
    expected = int(1)
    loader.close()
    assert got == expected


def test_close():
    """
    check if close method works as intended and closes file
    """
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
    loader.close()
    got = loader.h5_file.__bool__()
    expected = False
    assert got is expected


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
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
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


def test_init_incompatible_conditions():
    """
    check if the initialisation works as expected and raises an error when
    directories to ungrouped files is given but grouped variable is set to
    True
    """
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"
    with pytest.raises(IndexError) as execinfo:
        H5FileLoader(dir_path=dir_path, name=name, grouped=True)
    msg = " ".join(execinfo.value.args[0].split())
    assert "index out of range" in msg


def test_get_data_incompatible_args():
    """
    check if the get_data method works as expected and raises an error when
    data is ungrouped but index is not an int
    """
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
    index = (0, 1)
    with pytest.raises(AssertionError):
        loader.get_data(index)


def test_get_data_incorrect_args():
    """
    check if the get_data method works as expected and raises an error when
    an incorrect data type is fed in
    """
    dir_path = "./data/test/h5/paired/test"
    name = "fixed_images"

    loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
    index = "abc"
    with pytest.raises(ValueError) as execinfo:
        loader.get_data(index)
    msg = " ".join(execinfo.value.args[0].split())
    assert "must be int, or tuple" in msg
