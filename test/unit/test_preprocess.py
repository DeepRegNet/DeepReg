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
from test.unit.util import is_equal_np, is_equal_tf

import numpy as np
import pytest
import tensorflow as tf

import deepreg.dataset
import deepreg.dataset.preprocess as preprocess


@pytest.mark.parametrize(
    ("moving_input_size", "fixed_input_size", "moving_image_size", "fixed_image_size"),
    [
        ((1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)),
        ((3, 4, 5), (4, 5, 6), (1, 2, 3), (2, 3, 4)),
        ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    ],
)
@pytest.mark.parametrize("labeled", [True, False])
def test_resize_inputs(
    moving_input_size: tuple,
    fixed_input_size: tuple,
    moving_image_size: tuple,
    fixed_image_size: tuple,
    labeled: bool,
):
    """
    Check return shapes.

    :param moving_input_size: input moving image/label shape
    :param fixed_input_size: input fixed image/label shape
    :param moving_image_size: output moving image/label shape
    :param fixed_image_size: output fixed image/label shape
    :param labeled: if data is labeled
    """
    num_indices = 2

    moving_image = tf.random.uniform(moving_input_size)
    fixed_image = tf.random.uniform(fixed_input_size)
    indices = tf.ones((num_indices,))
    inputs = dict(moving_image=moving_image, fixed_image=fixed_image, indices=indices)
    if labeled:
        moving_label = tf.random.uniform(moving_input_size)
        fixed_label = tf.random.uniform(fixed_input_size)
        inputs["moving_label"] = moving_label
        inputs["fixed_label"] = fixed_label

    outputs = preprocess.resize_inputs(inputs, moving_image_size, fixed_image_size)
    assert inputs["indices"].shape == outputs["indices"].shape
    for k in inputs:
        if k == "indices":
            assert outputs[k].shape == inputs[k].shape
            continue
        expected_shape = moving_image_size if "moving" in k else fixed_image_size
        assert outputs[k].shape == expected_shape


def test_random_transform_3d_get_config():
    """Check config values."""
    config = dict(
        moving_image_size=(1, 2, 3),
        fixed_image_size=(2, 3, 4),
        batch_size=3,
        name="TestRandomTransformation3D",
    )
    expected = {"trainable": False, "dtype": "float32", **config}
    transform = preprocess.RandomTransformation3D(**config)
    got = transform.get_config()

    assert got == expected


class TestRandomTransformation:
    """Test all functions of RandomTransformation class."""

    moving_image_size = (1, 2, 3)
    fixed_image_size = (2, 3, 4)
    batch_size = 2
    scale = 0.2
    num_indices = 3
    name = "TestTransformation"
    common_config = dict(
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        batch_size=batch_size,
        name=name,
    )
    extra_config_dict = dict(
        affine=dict(scale=0.2), ddf=dict(field_strength=0.2, low_res_size=(1, 2, 3))
    )
    layer_cls_dict = dict(
        affine=preprocess.RandomAffineTransform3D,
        ddf=preprocess.RandomDDFTransform3D,
    )

    def build_layer(self, name: str) -> preprocess.RandomTransformation3D:
        """
        Build a layer given the layer name.

        :param name: name of the layer
        :return: built layer object
        """
        config = {**self.common_config, **self.extra_config_dict[name]}  # type: ignore
        return self.layer_cls_dict[name](**config)

    @pytest.mark.parametrize("name", ["affine", "ddf"])
    def test_get_config(self, name: str):
        """
        Check config values.

        :param name: name of the layer
        """
        layer = self.build_layer(name)
        got = layer.get_config()
        expected = {
            "trainable": False,
            "dtype": "float32",
            **self.common_config,  # type: ignore
            **self.extra_config_dict[name],  # type: ignore
        }
        assert got == expected

    @pytest.mark.parametrize(
        ("name", "moving_param_shape", "fixed_param_shape"),
        [
            ("affine", (4, 3), (4, 3)),
            ("ddf", (*moving_image_size, 3), (*fixed_image_size, 3)),
        ],
    )
    def test_gen_transform_params(
        self, name: str, moving_param_shape: tuple, fixed_param_shape: tuple
    ):
        """
        Check return shapes and moving/fixed params should be different.

        :param name: name of the layer
        :param moving_param_shape: params shape for moving image/label
        :param fixed_param_shape: params shape for fixed image/label
        """
        layer = self.build_layer(name)
        moving, fixed = layer.gen_transform_params()
        assert moving.shape == (self.batch_size, *moving_param_shape)
        assert fixed.shape == (self.batch_size, *fixed_param_shape)
        assert not is_equal_np(moving, fixed)

    @pytest.mark.parametrize("name", ["affine", "ddf"])
    def test_transform(self, name: str):
        """
        Check return shapes.

        :param name: name of the layer
        """
        layer = self.build_layer(name)
        moving_image = tf.random.uniform(
            shape=(self.batch_size, *self.moving_image_size)
        )
        moving_params, _ = layer.gen_transform_params()
        transformed = layer.transform(
            image=moving_image,
            grid_ref=layer.moving_grid_ref,
            params=moving_params,
        )
        assert transformed.shape == moving_image.shape

    @pytest.mark.parametrize("name", ["affine", "ddf"])
    @pytest.mark.parametrize("labeled", [True, False])
    def test_call(self, name: str, labeled: bool):
        """
        Check return shapes.

        :param name: name of the layer
        :param labeled: if data is labeled
        """
        layer = self.build_layer(name)

        moving_shape = (self.batch_size, *self.moving_image_size)
        fixed_shape = (self.batch_size, *self.fixed_image_size)
        moving_image = tf.random.uniform(moving_shape)
        fixed_image = tf.random.uniform(fixed_shape)
        indices = tf.ones((self.batch_size, self.num_indices))
        inputs = dict(
            moving_image=moving_image, fixed_image=fixed_image, indices=indices
        )
        if labeled:
            moving_label = tf.random.uniform(moving_shape)
            fixed_label = tf.random.uniform(fixed_shape)
            inputs["moving_label"] = moving_label
            inputs["fixed_label"] = fixed_label

        outputs = layer.call(inputs)
        for k in inputs:
            assert outputs[k].shape == inputs[k].shape


def test_random_transform_generator():
    """
    Test random_transform_generator by confirming that it generates
    appropriate solutions and output sizes for seeded examples.
    """
    # Check shapes are correct Batch Size = 1 - Pass
    batch_size = 1
    transforms = deepreg.dataset.preprocess.gen_rand_affine_transform(batch_size, 0)
    assert transforms.shape == (batch_size, 4, 3)

    # Check numerical outputs are correct for a given seed - Pass
    batch_size = 1
    scale = 0.1
    seed = 0
    expected = tf.constant(
        np.array(
            [
                [
                    [9.4661278e-01, -3.8267835e-03, 3.6934228e-03],
                    [5.5613145e-03, 9.8034811e-01, -1.8044969e-02],
                    [1.9651605e-04, 1.4576728e-02, 9.6243286e-01],
                    [-2.5107686e-03, 1.9579126e-02, -1.2195010e-02],
                ]
            ],
            dtype=np.float32,
        )
    )  # shape = (1, 4, 3)
    got = deepreg.dataset.preprocess.gen_rand_affine_transform(
        batch_size=batch_size, scale=scale, seed=seed
    )
    assert is_equal_tf(got, expected)
