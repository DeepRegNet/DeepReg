"""
Tests functionality of the PairedDataLoader
"""
from os.path import join

import numpy as np
import pytest

from deepreg.dataset.loader.h5_loader import H5FileLoader
from deepreg.dataset.loader.nifti_loader import NiftiFileLoader
from deepreg.dataset.loader.paired_loader import PairedDataLoader

# assign values to input vars
moving_image_shape = (64, 64, 60)
fixed_image_shape = (32, 32, 60)

FileLoaderDict = dict(nifti=NiftiFileLoader, h5=H5FileLoader)
DataPaths = dict(nifti="data/test/nifti/paired", h5="data/test/h5/paired")


def test_init():
    """
    Check that data loader __init__() method is correct:
    """

    for key_file_loader, file_loader in FileLoaderDict.items():
        data_dir_path = [
            join(DataPaths[key_file_loader], "train"),
            join(DataPaths[key_file_loader], "test"),
        ]
        common_args = dict(
            file_loader=file_loader, labeled=True, sample_label="all", seed=None
        )
        data_loader = PairedDataLoader(
            data_dir_paths=data_dir_path,
            fixed_image_shape=fixed_image_shape,
            moving_image_shape=moving_image_shape,
            **common_args,
        )

        # Check that file loaders are initialized correctly
        file_loader_method = file_loader(
            dir_paths=data_dir_path, name="moving_images", grouped=False
        )
        assert isinstance(data_loader.loader_moving_image, type(file_loader_method))
        assert isinstance(data_loader.loader_fixed_image, type(file_loader_method))
        assert isinstance(data_loader.loader_moving_label, type(file_loader_method))
        assert isinstance(data_loader.loader_fixed_label, type(file_loader_method))

        data_loader.close()

        # Check the data_dir_path variable assertion error.
        data_dir_path_int = [0, "1", 2, 3]
        with pytest.raises(AssertionError):
            PairedDataLoader(
                data_dir_paths=data_dir_path_int,
                fixed_image_shape=fixed_image_shape,
                moving_image_shape=moving_image_shape,
                **common_args,
            )


def test_validate_data_files_label():
    """
    Test the validate_data_files functions
    that looks for inconsistencies in the fixed/moving image and label lists.
    If there is any issue it will raise an error, otherwise it returns None.
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:
            data_dir_path = [join(DataPaths[key_file_loader], split)]
            common_args = dict(
                file_loader=file_loader,
                labeled=True,
                sample_label="all",
                seed=None if split == "train" else 0,
            )

            data_loader = PairedDataLoader(
                data_dir_paths=data_dir_path,
                fixed_image_shape=fixed_image_shape,
                moving_image_shape=moving_image_shape,
                **common_args,
            )

            assert data_loader.validate_data_files() is None
            data_loader.close()


def test_sample_index_generator():
    """
    Test to check the randomness and deterministic index generator
    for train/test respectively.
    """

    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:
            data_dir_path = [join(DataPaths[key_file_loader], split)]
            indices_to_compare = []

            for seed in [0, 1, 0]:
                data_loader = PairedDataLoader(
                    data_dir_paths=data_dir_path,
                    fixed_image_shape=fixed_image_shape,
                    moving_image_shape=moving_image_shape,
                    file_loader=file_loader,
                    labeled=True,
                    sample_label="all",
                    seed=seed,
                )

                data_indices = []
                for (
                    moving_index,
                    fixed_index,
                    indices,
                ) in data_loader.sample_index_generator():
                    assert isinstance(moving_index, int)
                    assert isinstance(fixed_index, int)
                    assert isinstance(indices, list)
                    assert moving_index == fixed_index
                    data_indices += indices

                indices_to_compare.append(data_indices)
                data_loader.close()

            if data_loader.num_images > 1:
                # test different seeds give different indices
                assert not (np.allclose(indices_to_compare[0], indices_to_compare[1]))
                # test same seeds give the same indices
                assert np.allclose(indices_to_compare[0], indices_to_compare[2])


def test_close():
    """
    Test the close function. Only needed for H5 data loaders for now.
    Since fixed/moving loaders are the same for
    unpaired data loader, only need to test the moving.
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:

            data_dir_path = [join(DataPaths[key_file_loader], split)]
            common_args = dict(
                file_loader=file_loader,
                labeled=True,
                sample_label="all",
                seed=None if split == "train" else 0,
            )

            data_loader = PairedDataLoader(
                data_dir_paths=data_dir_path,
                fixed_image_shape=fixed_image_shape,
                moving_image_shape=moving_image_shape,
                **common_args,
            )

            if key_file_loader == "h5":
                data_loader.close()
                for f in data_loader.loader_moving_image.h5_files.values():
                    assert not f.__bool__()
