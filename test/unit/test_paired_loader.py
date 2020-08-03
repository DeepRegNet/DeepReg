"""
Tests functionality of the PairedDataLoader
"""
import pytest

from deepreg.dataset.loader.h5_loader import H5FileLoader
from deepreg.dataset.loader.paired_loader import PairedDataLoader

# assign values to input vars
data_dir_path = "./data/test/h5/paired/test"
sample_label = "sample"
seed = 1
moving_image_shape_arr = (8, 8, 8)
fixed_image_shape_arr = (8, 8, 8)


# in __init__: seed needs var spec, and sample_label should be optional
def test_init_sufficient_args():
    """
    check if init method of loader returns any errors when all required
    arguments given
    """

    loader = PairedDataLoader(
        file_loader=H5FileLoader,
        data_dir_path=data_dir_path,
        labeled=True,
        sample_label=sample_label,
        moving_image_shape=moving_image_shape_arr,
        fixed_image_shape=fixed_image_shape_arr,
        seed=seed,
    )
    loader.__init__(
        file_loader=H5FileLoader,
        data_dir_path=data_dir_path,
        labeled=True,
        sample_label=sample_label,
        moving_image_shape=moving_image_shape_arr,
        fixed_image_shape=fixed_image_shape_arr,
        seed=seed,
    )
    loader.close()


def test_init_num_images():
    """
    check init reads expected number of image pairs from given data path
    """

    loader = PairedDataLoader(
        file_loader=H5FileLoader,
        data_dir_path=data_dir_path,
        labeled=True,
        sample_label=sample_label,
        moving_image_shape=moving_image_shape_arr,
        fixed_image_shape=fixed_image_shape_arr,
        seed=seed,
    )
    got = loader.num_images
    expected = 1
    loader.close()
    assert got == expected


def test_file_loader_init():
    """
    check file loader is correctly called in __init__:
    """

    loader = PairedDataLoader(
        file_loader=H5FileLoader,
        data_dir_path=data_dir_path,
        labeled=True,
        sample_label=sample_label,
        moving_image_shape=moving_image_shape_arr,
        fixed_image_shape=fixed_image_shape_arr,
        seed=seed,
    )
    file_loader = H5FileLoader(
        dir_path=data_dir_path, name="moving_images", grouped=False
    )

    expected = ["case000025.nii.gz"]

    loader_got = loader.loader_moving_image.get_data_ids()
    file_loader_got = file_loader.get_data_ids()
    loader.close()
    file_loader.close()
    assert loader_got == expected, "paired_loader has loaded incorrect moving image"
    assert loader_got == file_loader_got, "paired_loader incorrectly calling h5_loader"


def test_validate_data_files_label():
    """
    check validate_data_files throws exception when moving and fixed label IDs vary
    """

    loader = PairedDataLoader(
        file_loader=H5FileLoader,
        data_dir_path=data_dir_path,
        labeled=True,
        sample_label=sample_label,
        moving_image_shape=moving_image_shape_arr,
        fixed_image_shape=fixed_image_shape_arr,
        seed=seed,
    )

    # alter a data ID to cause error
    loader.loader_moving_label.data_keys = "foo"
    with pytest.raises(ValueError) as err_info:
        PairedDataLoader.validate_data_files(loader)
    loader.close()
    assert "two lists are not identical" in str(err_info.value)


def test_sample_index_generator():
    """
    check image index is expected value and format
    """

    loader = PairedDataLoader(
        file_loader=H5FileLoader,
        data_dir_path=data_dir_path,
        labeled=True,
        sample_label=sample_label,
        moving_image_shape=moving_image_shape_arr,
        fixed_image_shape=fixed_image_shape_arr,
        seed=seed,
    )

    expected = (0, 0, [0])
    image_index = PairedDataLoader.sample_index_generator(loader)
    got = next(image_index)
    loader.close()
    assert expected == got
