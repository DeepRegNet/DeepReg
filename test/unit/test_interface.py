# coding=utf-8

"""
Tests for deepreg/dataset/loader/interface.py
"""
import numpy as np
import pytest

from deepreg.dataset.loader.interface import (
    AbstractPairedDataLoader,
    AbstractUnpairedDataLoader,
    DataLoader,
    FileLoader,
    GeneratorDataLoader,
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


def test_generator_data_loader(caplog):
    """
    Test the functions in GeneratorDataLoader
    """
    generator = GeneratorDataLoader(labeled=True, num_indices=1, sample_label="all")

    # test properties
    assert generator.loader_moving_image is None
    assert generator.loader_moving_image is None
    assert generator.loader_moving_image is None
    assert generator.loader_moving_image is None

    # not implemented properties / functions
    with pytest.raises(NotImplementedError):
        generator.sample_index_generator()

    # implemented functions
    # test get_Dataset
    dummy_array = np.random.random(size=(100, 100, 100)).astype(np.float32)

    # mock generator
    sequence = [
        dict(
            moving_image=dummy_array,
            fixed_image=dummy_array,
            moving_label=dummy_array,
            fixed_label=dummy_array,
            indices=[1],
        )
        for i in range(3)
    ]

    def mock_generator():
        for el in sequence:
            yield el

    # inputs, no error means passed
    generator.data_generator = mock_generator
    dataset = generator.get_dataset()

    # check output for data generator
    expected = dict(
        moving_image=dummy_array,
        fixed_image=dummy_array,
        moving_label=dummy_array,
        fixed_label=dummy_array,
        indices=[1],
    )
    for got in list(dataset.as_numpy_iterator()):
        assert all(np.array_equal(got[key], expected[key]) for key in expected.keys())

    # test validate_images_and_labels
    with pytest.raises(ValueError) as err_info:
        generator.validate_images_and_labels(
            fixed_image=None,
            moving_image=dummy_array,
            moving_label=None,
            fixed_label=None,
            image_indices=[1],
        )
    assert "moving image and fixed image must not be None" in str(err_info.value)
    with pytest.raises(ValueError) as err_info:
        generator.validate_images_and_labels(
            fixed_image=dummy_array,
            moving_image=dummy_array,
            moving_label=dummy_array,
            fixed_label=None,
            image_indices=[1],
        )
    assert "moving label and fixed label must be both None or non-None" in str(
        err_info.value
    )
    with pytest.raises(ValueError) as err_info:
        generator.validate_images_and_labels(
            fixed_image=dummy_array,
            moving_image=dummy_array + 1.0,
            moving_label=None,
            fixed_label=None,
            image_indices=[1],
        )
    assert "Sample [1]'s moving_image's values are not between [0, 1]" in str(
        err_info.value
    )
    with pytest.raises(ValueError) as err_info:
        generator.validate_images_and_labels(
            fixed_image=dummy_array,
            moving_image=np.random.random(size=(100, 100)),
            moving_label=None,
            fixed_label=None,
            image_indices=[1],
        )
    assert "Sample [1]'s moving_image' shape should have dimension of 3. " in str(
        err_info.value
    )
    with pytest.raises(ValueError) as err_info:
        generator.validate_images_and_labels(
            fixed_image=dummy_array,
            moving_image=dummy_array,
            moving_label=np.random.random(size=(100, 100)),
            fixed_label=dummy_array,
            image_indices=[1],
        )
    assert "Sample [1]'s moving_label' shape should have dimension of 3 or 4. " in str(
        err_info.value
    )
    with pytest.raises(ValueError) as err_info:
        generator.validate_images_and_labels(
            fixed_image=dummy_array,
            moving_image=dummy_array,
            moving_label=np.random.random(size=(100, 100, 100, 3)),
            fixed_label=np.random.random(size=(100, 100, 100, 4)),
            image_indices=[1],
        )
    assert (
        "Sample [1]'s moving image and fixed image have different numbers of labels."
        in str(err_info.value)
    )

    # warning
    generator.validate_images_and_labels(
        fixed_image=dummy_array,
        moving_image=dummy_array,
        moving_label=np.random.random(size=(100, 100, 90)),
        fixed_label=dummy_array,
        image_indices=[1],
    )
    assert "Sample [1]'s moving image and label have different shapes. " in caplog.text
    generator.validate_images_and_labels(
        fixed_image=dummy_array,
        moving_image=dummy_array,
        moving_label=dummy_array,
        fixed_label=np.random.random(size=(100, 100, 90)),
        image_indices=[1],
    )
    assert "Sample [1]'s fixed image and label have different shapes. " in caplog.text

    # test sample_image_label method
    # for unlabeled input data
    got = next(
        generator.sample_image_label(
            fixed_image=dummy_array,
            moving_image=dummy_array,
            moving_label=None,
            fixed_label=None,
            image_indices=[1],
        )
    )
    expected = dict(
        moving_image=dummy_array,
        fixed_image=dummy_array,
        indices=np.asarray([1] + [-1], dtype=np.float32),
    )
    assert all(np.array_equal(got[key], expected[key]) for key in expected.keys())

    # for data with one label
    got = next(
        generator.sample_image_label(
            fixed_image=dummy_array,
            moving_image=dummy_array,
            moving_label=dummy_array,
            fixed_label=dummy_array,
            image_indices=[1],
        )
    )
    expected = dict(
        moving_image=dummy_array,
        fixed_image=dummy_array,
        moving_label=dummy_array,
        fixed_label=dummy_array,
        indices=np.asarray([1] + [0], dtype=np.float32),
    )
    assert all(np.array_equal(got[key], expected[key]) for key in expected.keys())

    # for data with multiple labels
    dummy_labels = np.random.random(size=(100, 100, 100, 3))
    got = generator.sample_image_label(
        fixed_image=dummy_array,
        moving_image=dummy_array,
        moving_label=dummy_labels,
        fixed_label=dummy_labels,
        image_indices=[1],
    )
    for label_index in range(dummy_labels.shape[3]):
        got_iter = next(got)
        expected = dict(
            moving_image=dummy_array,
            fixed_image=dummy_array,
            moving_label=dummy_labels[..., label_index],
            fixed_label=dummy_labels[..., label_index],
            indices=np.asarray([1] + [label_index], dtype=np.float32),
        )
        assert all(
            np.array_equal(got_iter[key], expected[key]) for key in expected.keys()
        )


def test_file_loader():
    """
    Test the functions in FileLoader
    """
    # init, no error means passed
    loader_grouped = FileLoader(
        dir_path=["/path/grouped_loader/"], name="grouped_loader", grouped=True
    )
    loader_ungrouped = FileLoader(
        dir_path="/path/ungrouped_loader/", name="ungrouped_loader", grouped=False
    )

    # not implemented properties / functions
    with pytest.raises(NotImplementedError):
        loader_grouped.set_data_structure()
        loader_grouped.set_group_structure()
        loader_grouped.get_data()
        loader_grouped.get_data_ids()
        loader_grouped.get_num_images()
        loader_grouped.close()

    # test grouped file loader functions
    assert loader_grouped.group_struct is None

    # create mock group structure with nested list
    loader_grouped.group_struct = [[1, 2], [3, 4], [5, 6]]
    assert loader_grouped.get_num_groups() == 3
    assert loader_grouped.get_num_images_per_group() == [2, 2, 2]

    # test ungrouped file loader
    with pytest.raises(AttributeError):
        loader_ungrouped.group_struct
    with pytest.raises(AssertionError):
        loader_ungrouped.get_num_groups()
    with pytest.raises(AssertionError):
        loader_ungrouped.get_num_images_per_group()
