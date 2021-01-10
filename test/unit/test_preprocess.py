"""
Tests for deepreg/dataset/preprocess.py in
pytest style

Some internals of the _gen_transform, _transform and
transform function, such as:
    - layer_util.random_transform_generator
    - layer_util.warp_grid
    - layer_util.resample
Are assumed working, and are tested separately in
test_layer_util.py; as such we just check output size here.
"""
import numpy as np
import pytest
import tensorflow as tf

import deepreg.dataset.preprocess as preprocess


@pytest.mark.parametrize(
    "input_size,moving_image_size,fixed_image_size",
    [
        ((2, 2, 2), (1, 3, 5), (2, 4, 6)),
        ((2, 2, 2), (1, 3, 5), (1, 3, 5)),
        ((4, 4, 4), (4, 4, 4), (2, 4, 6)),
    ],
)
def test_resize_inputs(input_size, moving_image_size, fixed_image_size):
    """
    Test resize_inputs by confirming that it generates
    appropriate output sizes on a simple test case.
    :param input_size: tuple
    :param moving_image_size: tuple
    :param fixed_image_size: tuple
    :return:
    """
    # labeled data - Pass
    moving_image = tf.ones(input_size)
    fixed_image = tf.ones(input_size)
    moving_label = tf.ones(input_size)
    fixed_label = tf.ones(input_size)
    indices = tf.ones((2,))

    inputs = dict(
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_label=moving_label,
        fixed_label=fixed_label,
        indices=indices,
    )
    outputs = preprocess.resize_inputs(
        inputs=inputs,
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
    )

    assert outputs["moving_image"].shape == moving_image_size
    assert outputs["fixed_image"].shape == fixed_image_size
    assert outputs["moving_label"].shape == moving_image_size
    assert outputs["fixed_label"].shape == fixed_image_size

    # unlabeled data - Pass
    moving_image = tf.ones(input_size)
    fixed_image = tf.ones(input_size)
    indices = tf.ones((2,))

    inputs = dict(moving_image=moving_image, fixed_image=fixed_image, indices=indices)
    outputs = preprocess.resize_inputs(
        inputs=inputs,
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
    )

    assert outputs["moving_image"].shape == moving_image_size
    assert outputs["fixed_image"].shape == fixed_image_size


class TestAffineTransformation3d:
    @pytest.mark.parametrize(
        "dims,batch_size,scale",
        [
            ((3, 3, 3), 2, 0.1),
            ((9, 9, 9), 1, 0.1),
            ((3, 3, 3), 10, 0.1),
        ],
    )
    def test__gen_transforms(self, dims, batch_size, scale):
        """
        Check return shapes.

        :param dims: tuple
        :param batch_size: int
        :param scale: float
        :return:
        """

        moving_image = np.random.uniform(size=dims)
        fixed_image = np.random.uniform(size=dims)

        affine_transform_3d = preprocess.AffineTransformation3D(
            moving_image.shape, fixed_image.shape, batch_size, scale
        )

        transforms = affine_transform_3d._gen_transform_params()
        assert len(transforms) == 2
        assert transforms[0].shape == (batch_size, 4, 3)
        assert transforms[1].shape == (batch_size, 4, 3)

    @pytest.mark.parametrize(
        "dims,batch_size,scale",
        [
            ((3, 3, 3), 2, 0.1),
            ((9, 9, 9), 1, 0.1),
            ((3, 3, 3), 10, 0.1),
        ],
    )
    def test__transform(self, dims, batch_size, scale):
        """
        Test _transform() by confirming that it generates
        appropriate transform output sizes.
        :param dims: tuple
        :param batch_size: int
        :param scale: int
        :return:
        """
        moving_image = np.random.uniform(size=dims)
        fixed_image = np.random.uniform(size=dims)

        moving_image_batched = np.float32(
            np.repeat(moving_image[np.newaxis, :, :, :], batch_size, axis=0)
        )
        fixed_image_batched = np.float32(
            np.repeat(fixed_image[np.newaxis, :, :, :], batch_size, axis=0)
        )

        affine_transform_3d = preprocess.AffineTransformation3D(
            moving_image.shape, fixed_image.shape, batch_size, scale
        )

        transforms = affine_transform_3d._gen_transform_params()
        assert len(transforms) == 2
        assert transforms[0].shape == (batch_size, 4, 3)
        assert transforms[1].shape == (batch_size, 4, 3)

        (
            moving_transforms,
            fixed_transforms,
        ) = affine_transform_3d._gen_transform_params()
        transformed_moving_image = affine_transform_3d._transform(
            moving_image_batched,
            affine_transform_3d._moving_grid_ref,
            moving_transforms,
        )
        assert transformed_moving_image.shape == moving_image_batched.shape

        transformed_fixed_image = affine_transform_3d._transform(
            fixed_image_batched, affine_transform_3d._fixed_grid_ref, fixed_transforms
        )
        assert transformed_fixed_image.shape == fixed_image_batched.shape

        assert not np.allclose(moving_transforms, fixed_transforms)

    @pytest.mark.parametrize(
        "dims,batch_size,scale",
        [
            ((3, 3, 3), 2, 0.1),
            ((9, 9, 9), 1, 0.1),
            ((3, 3, 3), 10, 0.1),
        ],
    )
    def test_transform(self, dims, batch_size, scale):
        """
        Test transform() by comfirming that it params
        images and labels only as necessary and when provided.
        :param dims: tuple
        :param batch_size: int
        :param scale: int
        :return:
        """
        # Common test setup.
        moving_image = np.random.uniform(size=dims)
        fixed_image = np.random.uniform(size=dims)

        affine_transform_3d = preprocess.AffineTransformation3D(
            moving_image.shape, fixed_image.shape, batch_size, scale
        )

        moving_image_batched = np.float32(
            np.repeat(moving_image[np.newaxis, :, :, :], batch_size, axis=0)
        )
        fixed_image_batched = np.float32(
            np.repeat(fixed_image[np.newaxis, :, :, :], batch_size, axis=0)
        )

        indices = range(batch_size)

        # Test with no labels provided.
        inputs = {
            "moving_image": moving_image_batched,
            "fixed_image": fixed_image_batched,
            "indices": indices,
        }

        outputs = affine_transform_3d.transform(inputs)

        assert outputs.get("moving_image") is not None
        assert outputs.get("fixed_image") is not None
        assert outputs.get("indices") is not None

        assert outputs.get("moving_image").shape == moving_image_batched.shape
        assert outputs.get("fixed_image").shape == fixed_image_batched.shape
        assert len(outputs.get("indices")) == len(indices)

        # Test with labels provided.
        moving_label = np.round(np.random.uniform(size=dims))
        fixed_label = np.round(np.random.uniform(size=dims))
        moving_label_batched = np.float32(
            np.repeat(moving_label[np.newaxis, :, :, :], batch_size, axis=0)
        )
        fixed_label_batched = np.float32(
            np.repeat(fixed_label[np.newaxis, :, :, :], batch_size, axis=0)
        )

        inputs = {
            "moving_image": moving_image_batched,
            "fixed_image": fixed_image_batched,
            "moving_label": moving_label_batched,
            "fixed_label": fixed_label_batched,
            "indices": indices,
        }

        outputs = affine_transform_3d.transform(inputs)

        assert outputs.get("moving_image") is not None
        assert outputs.get("fixed_image") is not None
        assert outputs.get("moving_label") is not None
        assert outputs.get("fixed_label") is not None
        assert outputs.get("indices") is not None

        assert outputs.get("moving_image").shape == moving_image_batched.shape
        assert outputs.get("fixed_image").shape == fixed_image_batched.shape
        assert outputs.get("moving_label").shape == moving_image_batched.shape
        assert outputs.get("fixed_label").shape == fixed_image_batched.shape
        assert len(outputs.get("indices")) == len(indices)


class TestFFDTransformation3d:
    @pytest.mark.parametrize(
        "fix_dims,mov_dims,batch_size,field_strength,low_res_size,raise_error",
        [
            ((9, 9, 9), (9, 9, 9), 2, 5, (3, 3, 3), False),
            ((9, 9, 9), (15, 15, 15), 2, 5, (3, 3, 3), False),
            ((9, 9, 9), (3, 3, 3), 2, 5, (9, 9, 9), True),
        ],
    )
    def test_init(
        self, fix_dims, mov_dims, batch_size, field_strength, low_res_size, raise_error
    ):
        """
        Test initialization of FFDTransformation3D class
        :param fix_dims: tuple
        :param mov_dims: tuple
        :param batch_size: int
        :param field_strength: int
        :param low_res_size: tuple
        :param raise_error: bool, True if the specified parameters will raise error
        :return:
        """
        moving_image = np.random.uniform(size=mov_dims)
        fixed_image = np.random.uniform(size=fix_dims)

        if raise_error:
            with pytest.raises(AssertionError):
                ddf_transform_3d = preprocess.FFDTransformation3D(
                    moving_image.shape,
                    fixed_image.shape,
                    batch_size,
                    field_strength,
                    low_res_size,
                )
        else:

            ddf_transform_3d = preprocess.FFDTransformation3D(
                moving_image.shape,
                fixed_image.shape,
                batch_size,
                field_strength,
                low_res_size,
            )

            assert ddf_transform_3d._moving_grid_ref.shape == mov_dims + (3,)
            assert ddf_transform_3d._fixed_grid_ref.shape == fix_dims + (3,)

    @pytest.mark.parametrize(
        "dims,batch_size,field_strength,low_res_size",
        [
            ((9, 9, 9), 2, 5, (3, 3, 3)),
            ((15, 15, 15), 2, 5, (3, 3, 3)),
            ((3, 3, 3), 2, 5, (3, 3, 3)),
        ],
    )
    def test__gen_transforms(self, dims, batch_size, field_strength, low_res_size):
        """
        Test _gen_transforms() by confirming that it generates
        appropriate transform output sizes.
        :param dims: tuple
        :param batch_size: int
        :param field_strength: int
        :param low_res_size: tuple
        """
        moving_image = np.random.uniform(size=dims)
        fixed_image = np.random.uniform(size=dims)

        ddf_transform_3d = preprocess.FFDTransformation3D(
            moving_image.shape,
            fixed_image.shape,
            batch_size,
            field_strength,
            low_res_size,
        )

        transforms = ddf_transform_3d._gen_transform_params()
        assert len(transforms) == 2
        assert transforms[0].shape == (batch_size,) + dims + (3,)
        assert transforms[1].shape == (batch_size,) + dims + (3,)

    @pytest.mark.parametrize(
        "dims,batch_size,field_strength,low_res_size",
        [
            ((9, 9, 9), 2, 5, (3, 3, 3)),
            ((9, 9, 9), 2, 5, (3, 3, 3)),
            ((9, 9, 9), 2, 5, (3, 3, 3)),
        ],
    )
    def test__transform(self, dims, batch_size, field_strength, low_res_size):
        """
        Test _transform() by confirming that it generates
        appropriate transform output sizes.
        :param dims: tuple
        :param batch_size: int
        :param field_strength: int
        :param low_res_size: tuple
        """
        moving_image = np.random.uniform(size=dims)
        fixed_image = np.random.uniform(size=dims)

        moving_image_batched = np.float32(
            np.repeat(moving_image[np.newaxis, :, :, :], batch_size, axis=0)
        )
        fixed_image_batched = np.float32(
            np.repeat(fixed_image[np.newaxis, :, :, :], batch_size, axis=0)
        )

        ddf_transform_3d = preprocess.FFDTransformation3D(
            moving_image.shape,
            fixed_image.shape,
            batch_size,
            field_strength,
            low_res_size,
        )

        moving_transforms, fixed_transforms = ddf_transform_3d._gen_transform_params()

        transformed_moving_image = ddf_transform_3d._transform(
            moving_image_batched, ddf_transform_3d._moving_grid_ref, moving_transforms
        )
        assert transformed_moving_image.shape == moving_image_batched.shape

        transformed_fixed_image = ddf_transform_3d._transform(
            fixed_image_batched, ddf_transform_3d._fixed_grid_ref, fixed_transforms
        )
        assert transformed_fixed_image.shape == fixed_image_batched.shape

        assert not np.allclose(moving_transforms, fixed_transforms)

    @pytest.mark.parametrize(
        "dims,batch_size,field_strength,low_res_size",
        [
            ((9, 9, 9), 2, 5, (3, 3, 3)),
            ((9, 9, 9), 2, 5, (3, 3, 3)),
            ((9, 9, 9), 2, 5, (3, 3, 3)),
        ],
    )
    def test_transform(self, dims, batch_size, field_strength, low_res_size):
        """
        Test transform() by confirming that it generates
        appropriate transform output sizes.
        :param dims: tuple
        :param batch_size: int
        :param field_strength: int
        :param low_res_size: tuple
        """
        # Common test setup.
        moving_image = np.random.uniform(size=dims)
        fixed_image = np.random.uniform(size=dims)

        ddf_transform_3d = preprocess.FFDTransformation3D(
            moving_image.shape,
            fixed_image.shape,
            batch_size,
            field_strength,
            low_res_size,
        )

        moving_image_batched = np.float32(
            np.repeat(moving_image[np.newaxis, :, :, :], batch_size, axis=0)
        )
        fixed_image_batched = np.float32(
            np.repeat(fixed_image[np.newaxis, :, :, :], batch_size, axis=0)
        )

        indices = range(batch_size)

        # Test with no labels provided.
        inputs = {
            "moving_image": moving_image_batched,
            "fixed_image": fixed_image_batched,
            "indices": indices,
        }

        outputs = ddf_transform_3d.transform(inputs)

        assert outputs.get("moving_image") is not None
        assert outputs.get("fixed_image") is not None
        assert outputs.get("indices") is not None

        assert outputs.get("moving_image").shape == moving_image_batched.shape
        assert outputs.get("fixed_image").shape == fixed_image_batched.shape
        assert len(outputs.get("indices")) == len(indices)

        # Test with labels provided.
        moving_label = np.round(np.random.uniform(size=dims))
        fixed_label = np.round(np.random.uniform(size=dims))
        moving_label_batched = np.float32(
            np.repeat(moving_label[np.newaxis, :, :, :], batch_size, axis=0)
        )
        fixed_label_batched = np.float32(
            np.repeat(fixed_label[np.newaxis, :, :, :], batch_size, axis=0)
        )

        inputs = {
            "moving_image": moving_image_batched,
            "fixed_image": fixed_image_batched,
            "moving_label": moving_label_batched,
            "fixed_label": fixed_label_batched,
            "indices": indices,
        }

        outputs = ddf_transform_3d.transform(inputs)

        assert outputs.get("moving_image") is not None
        assert outputs.get("fixed_image") is not None
        assert outputs.get("moving_label") is not None
        assert outputs.get("fixed_label") is not None
        assert outputs.get("indices") is not None

        assert outputs.get("moving_image").shape == moving_image_batched.shape
        assert outputs.get("fixed_image").shape == fixed_image_batched.shape
        assert outputs.get("moving_label").shape == moving_image_batched.shape
        assert outputs.get("fixed_label").shape == fixed_image_batched.shape
        assert len(outputs.get("indices")) == len(indices)
