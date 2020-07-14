# coding=utf-8

"""
Tests for deepreg/dataset/util.py in
pytest style
"""
import numpy as np

from deepreg.dataset.loader.h5_loader import H5FileLoader
from deepreg.dataset.loader.nifti_loader import NiftiFileLoader
from deepreg.dataset.loader.unpaired_loader import UnpairedDataLoader

FileLoaderDict = dict(nifti=NiftiFileLoader, h5=H5FileLoader)
DataPaths = dict(
    nifti="../../data/test/nifti/unpaired/train", h5="../../data/test/h5/unpaired/train"
)


def test_sample_index_generator():
    """
    Test to check the randomness and deterministic index generator for train/test respectively.
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:
            data_dir_path = DataPaths[key_file_loader]
            image_shape = (64, 64, 60)
            common_args = dict(
                file_loader=file_loader,
                labeled=True,
                sample_label="all",
                seed=None if split == "train" else 0,
            )

            data_loader = UnpairedDataLoader(
                data_dir_path=data_dir_path, image_shape=image_shape, **common_args
            )

            num_samples = data_loader.num_samples
            index_array = np.zeros((2 * num_samples, 2))
            for epoch in range(2):
                for it_indices, indices in enumerate(
                    data_loader.sample_index_generator()
                ):
                    image_indices = indices[2]
                    index_array[
                        2 * it_indices : 2 * (it_indices + 1), epoch
                    ] = image_indices
                    if it_indices >= (num_samples - 1):
                        break

            reference_index, moving_index, image_list = next(
                data_loader.sample_index_generator()
            )
            assert isinstance(reference_index, int)
            assert isinstance(moving_index, int)
            assert isinstance(image_list, list)
            if split == "train":
                assert np.allclose(index_array[..., 0], index_array[..., 1]) is False
            elif split == "test":
                assert np.allclose(index_array[..., 0], index_array[..., 1]) is True


def test_validate_data_files():
    """
    Test to check the randomness and deterministic index generator for train/test respectively.
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:

            data_dir_path = DataPaths[key_file_loader]
            image_shape = (64, 64, 60)
            common_args = dict(
                file_loader=file_loader,
                labeled=True,
                sample_label="all",
                seed=None if split == "train" else 0,
            )

            data_loader = UnpairedDataLoader(
                data_dir_path=data_dir_path, image_shape=image_shape, **common_args
            )

            assert data_loader.validate_data_files() is None


def test_close():
    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:

            data_dir_path = DataPaths[key_file_loader]
            image_shape = (64, 64, 60)
            common_args = dict(
                file_loader=file_loader,
                labeled=True,
                sample_label="all",
                seed=None if split == "train" else 0,
            )

            data_loader = UnpairedDataLoader(
                data_dir_path=data_dir_path, image_shape=image_shape, **common_args
            )

            if key_file_loader == "h5":
                data_loader.close()
                assert data_loader.loader_fixed_image.h5_file.__bool__() is False
                assert data_loader.loader_moving_image.h5_file.__bool__() is False
                assert data_loader.loader_fixed_label.h5_file.__bool__() is False
                assert data_loader.loader_moving_image.h5_file.__bool__() is False
            else:
                pass
