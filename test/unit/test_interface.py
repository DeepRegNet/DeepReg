# coding=utf-8

"""
Tests for deepreg/dataset/loader/interface.py
"""
from test.unit.util import is_equal_np

import numpy as np
import pytest

from deepreg.dataset.loader.interface import (
    AbstractPairedDataLoader,
    AbstractUnpairedDataLoader,
    DataLoader,
    FileLoader,
    GeneratorDataLoader,
)
from deepreg.dataset.loader.nifti_loader import NiftiFileLoader
from deepreg.dataset.loader.paired_loader import PairedDataLoader
from deepreg.dataset.loader.util import normalize_array


class TestDataLoader:
    @pytest.mark.parametrize(
        "labeled,num_indices,sample_label,seed",
        [
            (True, 1, "all", 0),
            (False, 1, "all", 0),
            (None, 1, "all", 0),
            (True, 1, "sample", 0),
            (True, 1, "all", 0),
            (True, 1, None, 0),
            (True, 1, "sample", None),
        ],
    )
    def test_init(self, labeled, num_indices, sample_label, seed):
        """
        Test init function of DataLoader class
        :param labeled: bool
        :param num_indices: int
        :param sample_label: str
        :param seed: float/int/None
        :return:
        """
        DataLoader(
            labeled=labeled,
            num_indices=num_indices,
            sample_label=sample_label,
            seed=seed,
        )

        data_loader = DataLoader(
            labeled=labeled,
            num_indices=num_indices,
            sample_label=sample_label,
            seed=seed,
        )

        with pytest.raises(NotImplementedError):
            data_loader.moving_image_shape
        with pytest.raises(NotImplementedError):
            data_loader.fixed_image_shape
        with pytest.raises(NotImplementedError):
            data_loader.num_samples
        with pytest.raises(NotImplementedError):
            data_loader.get_dataset()

        data_loader.close()

    @pytest.mark.parametrize(
        "labeled,moving_shape,fixed_shape,batch_size,data_augmentation",
        [
            (True, (9, 9, 9), (9, 9, 9), 1, {}),
            (
                True,
                (9, 9, 9),
                (15, 15, 15),
                1,
                {"data_augmentation": {"name": "affine"}},
            ),
            (
                True,
                (9, 9, 9),
                (15, 15, 15),
                1,
                {
                    "data_augmentation": [
                        {"name": "affine"},
                        {
                            "name": "ddf",
                            "field_strength": 1,
                            "low_res_size": (3, 3, 3),
                        },
                    ],
                },
            ),
        ],
    )
    def test_get_dataset_and_preprocess(
        self, labeled, moving_shape, fixed_shape, batch_size, data_augmentation
    ):
        """
        Test get_transforms() function. For that, an Abstract Data Loader is created
        only to set the moving  and fixed shapes that are used in get_transforms().
        Here we test that the get_transform() returns a function and the shape of
        the output of this function. See test_preprocess.py for more testing regarding
        the concrete params.

        :param labeled: bool
        :param moving_shape: tuple
        :param fixed_shape: tuple
        :param batch_size: total number of samples consumed per step, over all devices.
        :param data_augmentation: dict
        :return:
        """
        data_dir_path = [
            "data/test/nifti/paired/train",
            "data/test/nifti/paired/test",
        ]
        common_args = dict(
            file_loader=NiftiFileLoader, labeled=True, sample_label="all", seed=None
        )

        data_loader = PairedDataLoader(
            data_dir_paths=data_dir_path,
            fixed_image_shape=fixed_shape,
            moving_image_shape=moving_shape,
            **common_args,
        )

        dataset = data_loader.get_dataset_and_preprocess(
            training=True,
            batch_size=batch_size,
            repeat=True,
            shuffle_buffer_num_batch=1,
            **data_augmentation,
        )

        for outputs in dataset.take(1):
            assert (
                outputs["moving_image"].shape
                == (batch_size,) + data_loader.moving_image_shape
            )
            assert (
                outputs["fixed_image"].shape
                == (batch_size,) + data_loader.fixed_image_shape
            )
            assert (
                outputs["moving_label"].shape
                == (batch_size,) + data_loader.moving_image_shape
            )
            assert (
                outputs["fixed_label"].shape
                == (batch_size,) + data_loader.fixed_image_shape
            )


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
    assert "moving_image_shape and fixed_image_shape have length of three" in str(
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
    :param caplog: used to check warning message.
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
    # for unlabeled data
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

    # check dataset output
    expected = dict(
        moving_image=dummy_array,
        fixed_image=dummy_array,
        moving_label=dummy_array,
        fixed_label=dummy_array,
        indices=[1],
    )
    for got in list(dataset.as_numpy_iterator()):
        assert all(is_equal_np(got[key], expected[key]) for key in expected.keys())

    # for unlabeled data
    generator_unlabeled = GeneratorDataLoader(
        labeled=False, num_indices=1, sample_label="all"
    )

    sequence = [
        dict(moving_image=dummy_array, fixed_image=dummy_array, indices=[1])
        for i in range(3)
    ]

    # inputs, no error means passed
    generator_unlabeled.data_generator = mock_generator
    dataset = generator_unlabeled.get_dataset()

    # check dataset output
    expected = dict(moving_image=dummy_array, fixed_image=dummy_array, indices=[1])
    for got in list(dataset.as_numpy_iterator()):
        assert all(is_equal_np(got[key], expected[key]) for key in expected.keys())

    # test data_generator
    # create mock data loader and sample index generator
    class MockDataLoader:
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def get_data(index):
            return dummy_array

    def mock_sample_index_generator():
        return [[[1], [1], [1]]]

    generator = GeneratorDataLoader(labeled=True, num_indices=1, sample_label="all")
    generator.sample_index_generator = mock_sample_index_generator
    generator.loader_moving_image = MockDataLoader
    generator.loader_fixed_image = MockDataLoader
    generator.loader_moving_label = MockDataLoader
    generator.loader_fixed_label = MockDataLoader

    # check data generator output
    got = next(generator.data_generator())

    expected = dict(
        moving_image=normalize_array(dummy_array),
        fixed_image=normalize_array(dummy_array),
        moving_label=dummy_array,
        fixed_label=dummy_array,
        indices=np.asarray([1] + [0], dtype=np.float32),
    )
    assert all(is_equal_np(got[key], expected[key]) for key in expected.keys())

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
    assert "Sample [1]'s moving_image' shape should be 3D" in str(err_info.value)
    with pytest.raises(ValueError) as err_info:
        generator.validate_images_and_labels(
            fixed_image=dummy_array,
            moving_image=dummy_array,
            moving_label=np.random.random(size=(100, 100)),
            fixed_label=dummy_array,
            image_indices=[1],
        )
    assert "Sample [1]'s moving_label' shape should be 3D or 4D. " in str(
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
    caplog.clear()  # clear previous log
    generator.validate_images_and_labels(
        fixed_image=dummy_array,
        moving_image=dummy_array,
        moving_label=np.random.random(size=(100, 100, 90)),
        fixed_label=dummy_array,
        image_indices=[1],
    )
    assert "Sample [1]'s moving image and label have different shapes. " in caplog.text
    caplog.clear()  # clear previous log
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
    assert all(is_equal_np(got[key], expected[key]) for key in expected.keys())

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
    assert all(is_equal_np(got[key], expected[key]) for key in expected.keys())

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
        assert all(is_equal_np(got_iter[key], expected[key]) for key in expected.keys())


def test_file_loader():
    """
    Test the functions in FileLoader
    """
    # init, no error means passed
    loader_grouped = FileLoader(
        dir_paths=["/path/grouped_loader/"], name="grouped_loader", grouped=True
    )
    loader_ungrouped = FileLoader(
        dir_paths=["/path/ungrouped_loader/"], name="ungrouped_loader", grouped=False
    )

    # init fails with repeated paths
    with pytest.raises(ValueError) as err_info:
        FileLoader(
            dir_paths=["/path/ungrouped_loader/", "/path/ungrouped_loader/"],
            name="ungrouped_loader",
            grouped=False,
        )
    assert "dir_paths have repeated elements" in str(err_info.value)

    # not implemented properties / functions
    with pytest.raises(NotImplementedError):
        loader_grouped.set_data_structure()
    with pytest.raises(NotImplementedError):
        loader_grouped.set_group_structure()
    with pytest.raises(NotImplementedError):
        loader_grouped.get_data(1)
    with pytest.raises(NotImplementedError):
        loader_grouped.get_data_ids()
    with pytest.raises(NotImplementedError):
        loader_grouped.get_num_images()
    with pytest.raises(NotImplementedError):
        loader_grouped.close()

    # test grouped file loader functions
    assert loader_grouped.group_struct is None

    # create mock group structure with nested list
    loader_grouped.group_struct = [[1, 2], [3, 4], [5, 6]]
    assert loader_grouped.get_num_groups() == 3
    assert loader_grouped.get_num_images_per_group() == [2, 2, 2]
    with pytest.raises(ValueError) as err_info:
        loader_grouped.group_struct = [[], [3, 4], [5, 6]]
        loader_grouped.get_num_images_per_group()
    assert "Groups of ID [0, 2, 2] are empty." in str(err_info.value)

    # test ungrouped file loader
    assert loader_ungrouped.group_struct is None
    with pytest.raises(AssertionError):
        loader_ungrouped.get_num_groups()
    with pytest.raises(AssertionError):
        loader_ungrouped.get_num_images_per_group()
