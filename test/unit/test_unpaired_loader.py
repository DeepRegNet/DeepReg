# coding=utf-8

"""
Tests for deepreg/dataset/util.py in
pytest style
"""
from os.path import join

import numpy as np

from deepreg.dataset.loader.h5_loader import H5FileLoader
from deepreg.dataset.loader.nifti_loader import NiftiFileLoader
from deepreg.dataset.loader.unpaired_loader import UnpairedDataLoader

FileLoaderDict = dict(nifti=NiftiFileLoader, h5=H5FileLoader)
DataPaths = dict(nifti="data/test/nifti/unpaired", h5="data/test/h5/unpaired")


def test_sample_index_generator():
    """
    Test to check the randomness and deterministic index generator
    for train/test respectively.
    """
    image_shape = (64, 64, 60)

    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:
            data_dir_path = [join(DataPaths[key_file_loader], split)]
            indices_to_compare = []

            for seed in [0, 1, 0]:
                data_loader = UnpairedDataLoader(
                    data_dir_paths=data_dir_path,
                    image_shape=image_shape,
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
                    data_indices += indices

                indices_to_compare.append(data_indices)

            # test different seeds give different indices
            assert not np.allclose(indices_to_compare[0], indices_to_compare[1])
            # test same seeds give the same indices
            assert np.allclose(indices_to_compare[0], indices_to_compare[2])


def test_validate_data_files():
    """
    Test the validate_data_files functions that looks for inconsistencies
    in the fixed/moving image and label lists.
    If there is any issue it will raise an error, otherwise it returns None.
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:
            data_dir_path = [join(DataPaths[key_file_loader], split)]
            image_shape = (64, 64, 60)
            common_args = dict(
                file_loader=file_loader,
                labeled=True,
                sample_label="all",
                seed=None if split == "train" else 0,
            )

            data_loader = UnpairedDataLoader(
                data_dir_paths=data_dir_path, image_shape=image_shape, **common_args
            )

            assert data_loader.validate_data_files() is None


def test_close():
    """
    Test the close function. Only needed for H5 data loaders for now.
    Since fixed/moving loaders are the same for
    unpaired data loader, only need to test the moving.
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:

            data_dir_path = [join(DataPaths[key_file_loader], split)]
            image_shape = (64, 64, 60)
            common_args = dict(
                file_loader=file_loader,
                labeled=True,
                sample_label="all",
                seed=None if split == "train" else 0,
            )

            data_loader = UnpairedDataLoader(
                data_dir_paths=data_dir_path, image_shape=image_shape, **common_args
            )

            if key_file_loader == "h5":
                data_loader.close()
                for f in data_loader.loader_moving_image.h5_files.values():
                    assert not f.__bool__()
