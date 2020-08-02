"""
Tests for deepreg/dataset/loader/grouped_loader.py in
pytest style
"""
from os.path import join

import numpy as np
import pytest

from deepreg.dataset.loader.grouped_loader import GroupedDataLoader
from deepreg.dataset.loader.h5_loader import H5FileLoader
from deepreg.dataset.loader.nifti_loader import NiftiFileLoader

FileLoaderDict = dict(nifti=NiftiFileLoader, h5=H5FileLoader)
nifti_path = join("data", "test", "nifti", "grouped")
h5_path = join("data", "test", "h5", "grouped")
DataPaths = dict(nifti=nifti_path, h5=h5_path)


def test_validate_data_files():
    """
    Test validate_data_files function looks for inconsistencies in the fixed/moving image and label lists.
    If there is any issue it will raise an error, otherwise it returns None.
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for train_split in ["train", "test"]:
            for labeled in ["True", "False", 1, 0]:
                data_dir_path = join(DataPaths[key_file_loader], train_split)
                image_shape = (64, 64, 60)
                common_args = dict(
                    file_loader=file_loader,
                    labeled=labeled,
                    sample_label="all",
                    intra_group_prob=1,
                    intra_group_option="forward",
                    sample_image_in_group=False,
                    seed=None if train_split == "train" else 0,
                )

                data_loader = GroupedDataLoader(
                    data_dir_path=data_dir_path, image_shape=image_shape, **common_args
                )

                assert data_loader.validate_data_files() is None


def test_num_intra_group_probs():
    """
    Test whether fewer than two groups throws an error for intra_group_prob < 1
     - data/test/h5/grouped/test contains only a single group
    """
    with pytest.raises(Exception) as e_info:
        data_dir_path = join(DataPaths["h5"], "test")
        image_shape = (64, 64, 60)
        common_args = dict(
            file_loader=H5FileLoader,
            labeled=True,
            sample_label="all",
            intra_group_prob=0.5,
            intra_group_option="forward",
            sample_image_in_group=False,
            seed=0,
        )
        data_loader = GroupedDataLoader(
            data_dir_path=data_dir_path, image_shape=image_shape, **common_args
        )
        data_loader.close()

        assert "we need at least two groups" in str(e_info.value)


def test_sample_indices_intra():
    """
    Test the number of samples and indices are correct for a single image sample
    """
    data_dir_path = join(DataPaths["h5"], "train")
    image_shape = (64, 64, 60)
    common_args = dict(
        file_loader=FileLoaderDict["h5"],
        labeled=True,
        sample_label="all",
        intra_group_prob=0.5,
        intra_group_option="forward",
        sample_image_in_group=True,
        seed=None,
    )
    data_loader = GroupedDataLoader(
        data_dir_path=data_dir_path, image_shape=image_shape, **common_args
    )

    assert data_loader.sample_indices is None
    assert data_loader._num_samples == 2


def test_group_indices_inter():
    """
    Test the number of samples and indices are correct when entire group is sampled
    """
    data_dir_path = join(DataPaths["h5"], "train")
    image_shape = (64, 64, 60)
    common_args = dict(
        file_loader=FileLoaderDict["h5"],
        labeled=True,
        sample_label="all",
        intra_group_prob=0,
        intra_group_option="forward",
        sample_image_in_group=False,
        seed=None,
    )
    data_loader = GroupedDataLoader(
        data_dir_path=data_dir_path, image_shape=image_shape, **common_args
    )

    ni = np.array(data_loader.num_images_per_group)
    num_samples = np.sum(ni) * (np.sum(ni) - 1) - sum(ni * (ni - 1))

    sample_indices = data_loader.sample_indices
    sample_indices.sort()
    unique_indices = list(set(sample_indices))
    unique_indices.sort()

    assert data_loader._num_samples == num_samples
    assert sample_indices == unique_indices


def test_group_indices_intra():
    """
    Test sample numbers and indices are correct when entire group is sampled
    """
    data_dir_path = join(DataPaths["h5"], "train")
    image_shape = (64, 64, 60)
    for group_option in ["forward", "backward", "unconstrained"]:
        common_args = dict(
            file_loader=FileLoaderDict["h5"],
            labeled=True,
            sample_label="all",
            intra_group_prob=1,
            intra_group_option=group_option,
            sample_image_in_group=False,
            seed=None,
        )
        data_loader = GroupedDataLoader(
            data_dir_path=data_dir_path, image_shape=image_shape, **common_args
        )

        ni = np.array(data_loader.num_images_per_group)
        num_samples = sample_count(ni, direction=group_option)

        sample_indices = data_loader.sample_indices
        sample_indices.sort()
        unique_indices = list(set(sample_indices))
        unique_indices.sort()

        assert data_loader._num_samples == num_samples
        assert sample_indices == unique_indices


def sample_count(ni, direction="forward"):
    # helper function calculates number of samples
    if direction == "unconstrained":
        sample_total = sum(ni * (ni - 1))
    else:
        sample_total = sum(ni * (ni - 1) / 2)
    return int(sample_total)


def test_sample_index_generator():
    """
    Test to check the randomness and deterministic index generator for train
    test not checked because dir contains only a single group of 2 images
    """
    image_shape = (64, 64, 60)

    for key_file_loader, file_loader in FileLoaderDict.items():
        data_dir_path = join(DataPaths[key_file_loader], "train")

        for direction in ["forward", "backward", "unconstrained"]:
            indices_to_compare = []

            for seed in [0, 1, 0]:
                data_loader = GroupedDataLoader(
                    data_dir_path=data_dir_path,
                    image_shape=image_shape,
                    file_loader=file_loader,
                    labeled=True,
                    sample_label="all",
                    intra_group_prob=1,
                    intra_group_option=direction,
                    sample_image_in_group=True,
                    seed=seed,
                )

                data_indices = []
                for (
                    moving_index,
                    fixed_index,
                    indices,
                ) in data_loader.sample_index_generator():
                    assert isinstance(moving_index, tuple)
                    assert isinstance(fixed_index, tuple)
                    assert isinstance(indices, list)
                    data_indices += indices

                data_loader.close()
                indices_to_compare.append(data_indices)

            # test different seeds give different indices
            assert np.allclose(indices_to_compare[0], indices_to_compare[1]) is False
            # test same seeds give the same indices
            assert np.allclose(indices_to_compare[0], indices_to_compare[2]) is True


def test_close():
    """
    Test the close function
    Since fixed and moving loaders are the same only need to test the moving
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:

            data_dir_path = join(DataPaths[key_file_loader], split)
            image_shape = (64, 64, 60)
            data_loader = GroupedDataLoader(
                data_dir_path=data_dir_path,
                image_shape=image_shape,
                file_loader=file_loader,
                labeled=True,
                sample_label="all",
                intra_group_prob=1,
                intra_group_option="forward",
                sample_image_in_group=True,
                seed=0,
            )

            if key_file_loader == "h5":
                data_loader.close()
                assert data_loader.loader_moving_image.h5_file.__bool__() is False
                assert data_loader.loader_moving_image.h5_file.__bool__() is False
