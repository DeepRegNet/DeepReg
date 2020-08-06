# coding=utf-8

"""
Tests for deepreg/dataset/loader/interface.py
"""
import pytest

from deepreg.dataset.loader.interface import (
    AbstractPairedDataLoader,
    AbstractUnpairedDataLoader,
    DataLoader,
)


def test_data_loader():
    """
    Test the functions in DataLoader
    """

    # init
    # inputs, no error means passed
    for labeled in [True, False, None]:
        DataLoader(labeled=labeled, num_indices=1, sample_label="all", seed=0)
    for sample_label in ["sample", "all", None]:
        DataLoader(labeled=True, num_indices=1, sample_label=sample_label, seed=0)
    for num_indices in [1]:
        DataLoader(labeled=True, num_indices=num_indices, sample_label="sample", seed=0)
    for seed in [0, None]:
        DataLoader(labeled=True, num_indices=1, sample_label="sample", seed=seed)

    # not implemented properties / functions
    data_loader = DataLoader(labeled=True, num_indices=1, sample_label="sample", seed=0)
    with pytest.raises(NotImplementedError):
        data_loader.moving_image_shape
    with pytest.raises(NotImplementedError):
        data_loader.fixed_image_shape
    with pytest.raises(NotImplementedError):
        data_loader.num_samples
    with pytest.raises(NotImplementedError):
        data_loader.get_dataset()

    # implemented functions
    # TODO test get_dataset_and_preprocess

    data_loader.close()


def test_abstract_paired_data_loader():
    """
    Test the functions in AbstractPairedDataLoader
    """
    moving_image_shape = (8, 8, 4)
    fixed_image_shape = (6, 6, 4)

    # test init invalid shape
    with pytest.raises(ValueError) as err_info:
        AbstractPairedDataLoader(
            moving_image_shape=(2, 2),
            fixed_image_shape=(3, 3),
            labeled=True,
            sample_label="sample",
        )
    assert "moving_image_shape and fixed_image_shape have to be length of three" in str(
        err_info.value
    )

    # test init valid shapes
    data_loader = AbstractPairedDataLoader(
        moving_image_shape=moving_image_shape,
        fixed_image_shape=fixed_image_shape,
        labeled=True,
        sample_label="sample",
    )

    # test properties
    assert data_loader.num_indices == 2
    assert data_loader.moving_image_shape == moving_image_shape
    assert data_loader.fixed_image_shape == fixed_image_shape
    assert data_loader.num_samples is None


def test_abstract_unpaired_data_loader():
    """
    Test the functions in AbstractUnpairedDataLoader
    """
    image_shape = (8, 8, 4)

    # test init invalid shape
    with pytest.raises(ValueError) as err_info:
        AbstractUnpairedDataLoader(
            image_shape=(2, 2), labeled=True, sample_label="sample"
        )
    assert "image_shape has to be length of three" in str(err_info.value)

    # test init valid shapes
    data_loader = AbstractUnpairedDataLoader(
        image_shape=image_shape, labeled=True, sample_label="sample"
    )

    # test properties
    assert data_loader.num_indices == 3
    assert data_loader.moving_image_shape == image_shape
    assert data_loader.fixed_image_shape == image_shape
    assert data_loader.num_samples is None
