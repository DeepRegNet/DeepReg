# coding=utf-8

"""
Tests for deepreg/dataset/loader/interface.py
"""
from test.unit.util import is_equal_np
from typing import Optional, Tuple

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


def get_arr(shape: Tuple = (2, 3, 4), seed: Optional[int] = None) -> np.ndarray:
    """
    Return a random array.

    :param shape: shape of array.
    :param seed: random seed.
    :return: random array.
    """
    np.random.seed(seed)
    return np.random.random(size=shape).astype(np.float32)


class TestGeneratorDataLoader:
    @pytest.mark.parametrize("labeled", [True, False])
    def test_get_labeled_dataset(self, labeled: bool):
        """
        Test get_dataset with data loader.

        :param labeled: labeled data or not.
        """
        sample = {
            "moving_image": get_arr(),
            "fixed_image": get_arr(),
            "indices": [1],
        }
        if labeled:
            sample = {
                "moving_label": get_arr(),
                "fixed_label": get_arr(),
                **sample,
            }

        def mock_gen():
            """Toy data generator."""
            for _ in range(3):
                yield sample

        loader = GeneratorDataLoader(labeled=labeled, num_indices=1, sample_label="all")
        loader.__setattr__("data_generator", mock_gen)
        dataset = loader.get_dataset()
        for got in dataset.as_numpy_iterator():
            assert all(is_equal_np(got[key], sample[key]) for key in sample.keys())

    @pytest.mark.parametrize("labeled", [True, False])
    def test_data_generator(self, labeled: bool):
        """
        Test data_generator()

        :param labeled: labeled data or not.
        """

        class MockDataLoader:
            """Toy data loader."""

            def __init__(self, seed: int):
                """
                Init.

                :param seed: random seed for numpy.
                :param kwargs: additional arguments.
                """
                self.seed = seed

            def get_data(self, index: int) -> np.ndarray:
                """
                Return the dummy array despite of the index.

                :param index: not used
                :return: dummy array.
                """
                assert isinstance(index, int)
                return get_arr(seed=self.seed)

        def mock_sample_index_generator():
            """Toy sample index generator."""
            return [[1, 1, [1]]]

        loader = GeneratorDataLoader(labeled=labeled, num_indices=1, sample_label="all")
        loader.__setattr__("sample_index_generator", mock_sample_index_generator)
        loader.loader_moving_image = MockDataLoader(seed=0)
        loader.loader_fixed_image = MockDataLoader(seed=1)
        if labeled:
            loader.loader_moving_label = MockDataLoader(seed=2)
            loader.loader_fixed_label = MockDataLoader(seed=3)

        # check data loader output
        got = next(loader.data_generator())

        expected = {
            "moving_image": normalize_array(get_arr(seed=0)),
            "fixed_image": normalize_array(get_arr(seed=1)),
            # 0 or -1 is the label index
            "indices": np.array([1, 0] if labeled else [1, -1], dtype=np.float32),
        }
        if labeled:
            expected = {
                "moving_label": get_arr(seed=2),
                "fixed_label": get_arr(seed=3),
                **expected,
            }
        assert all(is_equal_np(got[key], expected[key]) for key in expected.keys())

    def test_sample_index_generator(self):
        loader = GeneratorDataLoader(labeled=True, num_indices=1, sample_label="all")
        with pytest.raises(NotImplementedError):
            loader.sample_index_generator()

    @pytest.mark.parametrize(
        (
            "moving_image_shape",
            "fixed_image_shape",
            "moving_label_shape",
            "fixed_label_shape",
            "err_msg",
        ),
        [
            (
                None,
                (10, 10, 10),
                (10, 10, 10),
                (10, 10, 10),
                "moving image and fixed image must not be None",
            ),
            (
                (10, 10, 10),
                None,
                (10, 10, 10),
                (10, 10, 10),
                "moving image and fixed image must not be None",
            ),
            (
                (10, 10, 10),
                (10, 10, 10),
                None,
                (10, 10, 10),
                "moving label and fixed label must be both None or non-None",
            ),
            (
                (10, 10, 10),
                (10, 10, 10),
                (10, 10, 10),
                None,
                "moving label and fixed label must be both None or non-None",
            ),
            (
                (10, 10),
                (10, 10, 10),
                (10, 10, 10),
                (10, 10, 10),
                "Sample [1]'s moving_image's shape should be 3D",
            ),
            (
                (10, 10, 10),
                (10, 10),
                (10, 10, 10),
                (10, 10, 10),
                "Sample [1]'s fixed_image's shape should be 3D",
            ),
            (
                (10, 10, 10),
                (10, 10, 10),
                (10, 10),
                (10, 10, 10),
                "Sample [1]'s moving_label's shape should be 3D or 4D.",
            ),
            (
                (10, 10, 10),
                (10, 10, 10),
                (10, 10, 10),
                (10, 10),
                "Sample [1]'s fixed_label's shape should be 3D or 4D.",
            ),
            (
                (10, 10, 10),
                (10, 10, 10),
                (10, 10, 10, 2),
                (10, 10, 10, 3),
                "Sample [1]'s moving image and fixed image "
                "have different numbers of labels.",
            ),
        ],
    )
    def test_validate_images_and_labels(
        self,
        moving_image_shape: Optional[Tuple],
        fixed_image_shape: Optional[Tuple],
        moving_label_shape: Optional[Tuple],
        fixed_label_shape: Optional[Tuple],
        err_msg: str,
    ):
        """
        Test error messages.

        :param moving_image_shape: None or tuple.
        :param fixed_image_shape: None or tuple.
        :param moving_label_shape: None or tuple.
        :param fixed_label_shape: None or tuple.
        :param err_msg: message.
        """
        moving_image = None
        fixed_image = None
        moving_label = None
        fixed_label = None
        if moving_image_shape:
            moving_image = get_arr(shape=moving_image_shape)
        if fixed_image_shape:
            fixed_image = get_arr(shape=fixed_image_shape)
        if moving_label_shape:
            moving_label = get_arr(shape=moving_label_shape)
        if fixed_label_shape:
            fixed_label = get_arr(shape=fixed_label_shape)
        loader = GeneratorDataLoader(labeled=True, num_indices=1, sample_label="all")
        with pytest.raises(ValueError) as err_info:
            loader.validate_images_and_labels(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_label=moving_label,
                fixed_label=fixed_label,
                image_indices=[1],
            )
        assert err_msg in str(err_info.value)

    @pytest.mark.parametrize("option", [0, 1, 2, 3])
    def test_validate_images_and_labels_range(self, option: int):
        """
        Test error messages related to input range.

        :param option: control which image to modify
        """
        option_to_name = {
            0: "moving_image",
            1: "fixed_image",
            2: "moving_label",
            3: "fixed_label",
        }
        input = {
            "moving_image": get_arr(),
            "fixed_image": get_arr(),
            "moving_label": get_arr(),
            "fixed_label": get_arr(),
        }
        name = option_to_name[option]
        input[name] += 1
        err_msg = f"Sample [1]'s {name}'s values are not between [0, 1]"

        loader = GeneratorDataLoader(labeled=True, num_indices=1, sample_label="all")
        with pytest.raises(ValueError) as err_info:
            loader.validate_images_and_labels(
                image_indices=[1],
                **input,
            )
        assert err_msg in str(err_info.value)

    def test_sample_image_label_unlabeled(self):
        """Test sample_image_label in unlabeled case."""
        loader = GeneratorDataLoader(labeled=False, num_indices=1, sample_label="all")
        got = next(
            loader.sample_image_label(
                moving_image=get_arr(seed=0),
                fixed_image=get_arr(seed=1),
                moving_label=None,
                fixed_label=None,
                image_indices=[1],
            )
        )
        expected = dict(
            moving_image=get_arr(seed=0),
            fixed_image=get_arr(seed=1),
            indices=np.asarray([1, -1], dtype=np.float32),
        )
        assert all(is_equal_np(got[key], expected[key]) for key in expected.keys())

    @pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 4, 1)])
    def test_sample_image_label_one_label(self, shape: Tuple):
        """
        Test sample_image_label in labeled case with one label.

        :param shape: shape of the label.
        """
        loader = GeneratorDataLoader(labeled=True, num_indices=1, sample_label="all")
        got = next(
            loader.sample_image_label(
                moving_image=get_arr(shape=shape[:3], seed=0),
                fixed_image=get_arr(shape=shape[:3], seed=1),
                moving_label=get_arr(shape=shape, seed=2),
                fixed_label=get_arr(shape=shape, seed=3),
                image_indices=[1],
            )
        )
        expected = dict(
            moving_image=get_arr(shape=shape[:3], seed=0),
            fixed_image=get_arr(shape=shape[:3], seed=1),
            moving_label=get_arr(shape=shape[:3], seed=2),
            fixed_label=get_arr(shape=shape[:3], seed=3),
            indices=np.asarray([1, 0], dtype=np.float32),
        )
        assert all(is_equal_np(got[key], expected[key]) for key in expected.keys())

    def test_sample_image_label_multiple_labels(self):
        """Test sample_image_label in labeled case with multiple labels."""
        loader = GeneratorDataLoader(labeled=True, num_indices=1, sample_label="all")
        shape = (2, 3, 4, 5)
        got_iter = loader.sample_image_label(
            moving_image=get_arr(shape=shape[:3], seed=0),
            fixed_image=get_arr(shape=shape[:3], seed=1),
            moving_label=get_arr(shape=shape, seed=2),
            fixed_label=get_arr(shape=shape, seed=3),
            image_indices=[1],
        )
        moving_label = get_arr(shape=shape, seed=2)
        fixed_label = get_arr(shape=shape, seed=3)
        for i in range(shape[-1]):
            got = next(got_iter)
            expected = dict(
                moving_image=get_arr(shape=shape[:3], seed=0),
                fixed_image=get_arr(shape=shape[:3], seed=1),
                moving_label=moving_label[:, :, :, i],
                fixed_label=fixed_label[:, :, :, i],
                indices=np.asarray([1, i], dtype=np.float32),
            )
            assert all(is_equal_np(got[key], expected[key]) for key in expected.keys())


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
