"""
Tests for deepreg/dataset/loader/grouped_loader.py in
pytest style
"""
from os.path import join
from typing import List

import numpy as np
import pytest

from deepreg.dataset.loader.grouped_loader import GroupedDataLoader
from deepreg.dataset.loader.h5_loader import H5FileLoader
from deepreg.dataset.loader.nifti_loader import NiftiFileLoader

FileLoaderDict = dict(nifti=NiftiFileLoader, h5=H5FileLoader)
DataPaths = dict(nifti="data/test/nifti/grouped", h5="data/test/h5/grouped")
image_shape = (64, 64, 60)


def sample_count(ni: List[int], direction: str) -> int:
    """
    Count number of samples.

    :param ni: list, each element correspond to the number of images per group
    :param direction: unconstrained/forward/backward
    :return: number of samples in total
    """
    arr = np.array(ni)
    if direction == "unconstrained":
        sample_total = np.sum(arr * (arr - 1))
    else:
        sample_total = np.sum(arr * (arr - 1) / 2)
    return int(sample_total)


def test_init():
    """
    Test exceptions with appropriate messages and counts samples correctly
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for train_split in ["test", "train"]:
            for prob in [0, 0.5, 1]:
                for sample_in_group in [True, False]:
                    data_dir_paths = [join(DataPaths[key_file_loader], train_split)]
                    common_args = dict(
                        file_loader=file_loader,
                        labeled=True,
                        sample_label="all",
                        intra_group_prob=prob,
                        intra_group_option="forward",
                        sample_image_in_group=sample_in_group,
                        seed=None,
                    )
                    if train_split == "test" and prob < 1:
                        # sample with fewer than 2 groups.
                        # In "test" we only have one group
                        with pytest.raises(ValueError) as err_info:
                            data_loader = GroupedDataLoader(
                                data_dir_paths=data_dir_paths,
                                image_shape=image_shape,
                                **common_args,
                            )
                            data_loader.close()
                        assert "we need at least two groups" in str(err_info.value)

                    elif train_split == "train" and sample_in_group is True:
                        # ensure sample count is accurate
                        # (only for train dir, test dir uses same logic)
                        data_loader = GroupedDataLoader(
                            data_dir_paths=data_dir_paths,
                            image_shape=image_shape,
                            **common_args,
                        )
                        assert data_loader.sample_indices is None
                        assert data_loader._num_samples == 2
                        data_loader.close()

                    elif sample_in_group is False and 0 < prob < 1:
                        # specifying conflicting intra/inter group parameters
                        with pytest.raises(ValueError) as err_info:
                            data_loader = GroupedDataLoader(
                                data_dir_paths=data_dir_paths,
                                image_shape=image_shape,
                                **common_args,
                            )
                            data_loader.close()
                        assert "Mixing intra and inter groups is not supported" in str(
                            err_info.value
                        )


def test_validate_data_files():
    """
    Test validate_data_files function looks for inconsistencies
     in the fixed/moving image and label lists.
    If there is any issue it will raise an error, otherwise it returns None.
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for train_split in ["train", "test"]:
            for labeled in [True, False]:
                data_dir_paths = [join(DataPaths[key_file_loader], train_split)]
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
                    data_dir_paths=data_dir_paths,
                    image_shape=image_shape,
                    **common_args,
                )

                assert data_loader.validate_data_files() is None


def test_get_inter_sample_indices():
    """
    Test all possible intergroup sampling indices are correctly calculated
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        data_dir_paths = [join(DataPaths[key_file_loader], "train")]
        common_args = dict(
            file_loader=file_loader,
            labeled=True,
            sample_label="all",
            intra_group_prob=0,
            intra_group_option="forward",
            sample_image_in_group=False,
            seed=None,
        )
        data_loader = GroupedDataLoader(
            data_dir_paths=data_dir_paths, image_shape=image_shape, **common_args
        )

        ni = np.array(data_loader.num_images_per_group)
        num_samples = np.sum(ni) * (np.sum(ni) - 1) - sum(ni * (ni - 1))

        sample_indices = data_loader.sample_indices
        sample_indices.sort()
        unique_indices = list(set(sample_indices))
        unique_indices.sort()

        assert data_loader._num_samples == num_samples
        assert sample_indices == unique_indices


def test_get_intra_sample_indices():
    """
    Test all possible intragroup sampling indices are correctly calculated
    Ensure exception is thrown for unsupported group_option
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:
            data_dir_paths = [join(DataPaths[key_file_loader], split)]
            common_args = dict(
                file_loader=file_loader,
                labeled=True,
                sample_label="all",
                intra_group_prob=1,
                sample_image_in_group=False,
                seed=None,
            )
            # test feasible intra_group_option
            for intra_group_option in ["forward", "backward", "unconstrained"]:
                data_loader = GroupedDataLoader(
                    data_dir_paths=data_dir_paths,
                    image_shape=image_shape,
                    intra_group_option=intra_group_option,
                    **common_args,
                )

                ni = data_loader.num_images_per_group
                num_samples = sample_count(ni, intra_group_option)

                sample_indices = data_loader.sample_indices
                sample_indices.sort()
                unique_indices = list(set(sample_indices))
                unique_indices.sort()

                # test all possible indices are generated
                assert data_loader._num_samples == num_samples
                assert sample_indices == unique_indices

            # test exception thrown for unsupported group option
            with pytest.raises(ValueError) as err_info:
                data_loader = GroupedDataLoader(
                    data_dir_paths=data_dir_paths,
                    image_shape=image_shape,
                    intra_group_option="wrong",
                    **common_args,
                )
                data_loader.close()
            assert "Unknown intra_group_option," in str(err_info.value)


def test_sample_index_generator():
    """
    Test to check the randomness and deterministic index generator for train
    Test dir not checked because it contains only a single group of 2 images
    """

    for key_file_loader, file_loader in FileLoaderDict.items():
        common_args = dict(
            image_shape=image_shape,
            data_dir_paths=[join(DataPaths[key_file_loader], "train")],
            file_loader=file_loader,
            labeled=True,
            sample_label="all",
        )

        # test feasible intra_group_option
        for sample_in_group in [False, True]:
            probs = [0, 0.5, 1] if sample_in_group else [0, 1]
            for prob in probs:
                for direction in ["forward", "backward", "unconstrained"]:
                    indices_to_compare = []

                    for seed in [0, 1, 0]:
                        data_loader = GroupedDataLoader(
                            intra_group_prob=prob,
                            intra_group_option=direction,
                            sample_image_in_group=sample_in_group,
                            seed=seed,
                            **common_args,
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
                    assert not np.allclose(indices_to_compare[0], indices_to_compare[1])
                    # test same seeds give the same indices
                    assert np.allclose(indices_to_compare[0], indices_to_compare[2])

        # test exception thrown for unsupported intra_group_option option
        data_loader = GroupedDataLoader(
            intra_group_prob=1,
            intra_group_option="wrong",
            sample_image_in_group=True,
            seed=0,
            **common_args,
        )
        with pytest.raises(ValueError) as err_info:
            next(data_loader.sample_index_generator())
        data_loader.close()
        assert "Unknown intra_group_option" in str(err_info.value)


def test_close():
    """
    Test the close function
    Since fixed and moving loaders are the same only need to test the moving
    """
    for key_file_loader, file_loader in FileLoaderDict.items():
        for split in ["train", "test"]:
            data_dir_paths = [join(DataPaths[key_file_loader], split)]

            data_loader = GroupedDataLoader(
                data_dir_paths=data_dir_paths,
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
                for f in data_loader.loader_moving_image.h5_files.values():
                    assert not f.__bool__()
