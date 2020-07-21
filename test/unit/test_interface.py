import unittest

from deepreg.dataset.loader.interface import (
    AbstractPairedDataLoader,
    AbstractUnpairedDataLoader,
    DataLoader,
)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """
        check if init method of loader returns any errors when all required
        arguments given
        """
        # test should not pass when wrong type
        self.testLoader = DataLoader(
            labeled=True, num_indices=1, sample_label="", seed=1
        )

    def test_NotImplementedError(self):
        with self.assertRaises(NotImplementedError):
            self.testLoader.moving_image_shape()
            self.testLoader.fixed_image_shape()
            self.testLoader.num_samples()
            self.testLoader.get_dataset()
            self.testLoader.get_dataset_and_preprocess(
                training=True, batch_size=2, repeat=True, shuffle_buffer_num_batch=2
            )


class TestAbstractPairedDataLoader(unittest.TestCase):
    def setUp(self):
        self.moving_image_shape = (1, 3, 4)
        self.fixed_image_shape = (1, 6, 8)
        self.testLoader = AbstractPairedDataLoader(
            moving_image_shape=self.moving_image_shape,
            fixed_image_shape=self.fixed_image_shape,
            labeled=True,
            sample_label="",
            seed=1,
        )

    @unittest.expectedFailure
    def test_init_fails(self):
        # invalid input shape
        self.testLoader2d = AbstractPairedDataLoader(
            moving_image_shape=(1, 3),
            fixed_image_shape=(1, 6),
            labeled=True,
            sample_label="",
            seed=1,
        )
        # argument already declared
        self.testLoader2d = AbstractPairedDataLoader(
            moving_image_shape=(1, 3, 4),
            fixed_image_shape=(1, 6, 8),
            labeled=True,
            sample_label="",
            seed=1,
            num_indices=2,
        )

        # missing argument
        self.testLoader2d = AbstractPairedDataLoader(
            moving_image_shape=(1, 3, 4),
            fixed_image_shape=(1, 3, 4),
            labeled=True,
            seed=1,
        )

    def test_property(self):
        assert self.testLoader.fixed_image_shape == self.fixed_image_shape
        assert self.testLoader.moving_image_shape == self.moving_image_shape
        assert self.testLoader.num_samples == self.testLoader.num_images


class TestAbstractUnpairedDataLoader(unittest.TestCase):
    def setUp(self):
        self.image_shape = (1, 3, 4)
        self.testLoader = AbstractUnpairedDataLoader(
            image_shape=self.image_shape, labeled=True, sample_label="", seed=1
        )

    @unittest.expectedFailure
    def test_init_fails(self):
        # invalid input shape
        self.testLoader2d = AbstractUnpairedDataLoader(
            image_shape=(1, 3), labeled=True, sample_label="", seed=1
        )
        # argument already declared
        self.testLoader2d = AbstractUnpairedDataLoader(
            image_shape=(1, 3, 4), labeled=True, sample_label="", seed=1, num_indices=2
        )

        # missing argument
        self.testLoader2d = AbstractUnpairedDataLoader(
            image_shape=(1, 3, 4), labeled=True, seed=1
        )

    def test_property(self):
        assert self.testLoader.fixed_image_shape == self.image_shape
        assert self.testLoader.moving_image_shape == self.image_shape
        assert self.testLoader.num_samples == self.testLoader._num_samples


# why return self.image and return self.sample ????
