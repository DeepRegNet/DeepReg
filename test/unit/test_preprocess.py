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
from test.unit.util import is_equal_np

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


def test_random_transform_3d_get_config():
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

    def build_layer(self, name):
        """
        Build a layer given the layer name.

        :param name: name of the layer
        :return: built layer object
        """
        config = {**self.common_config, **self.extra_config_dict[name]}
        return self.layer_cls_dict[name](**config)

    @pytest.mark.parametrize("name", ["affine", "ddf"])
    def test_get_config(self, name):
        """
        Check config values.

        :param name: name of the layer
        """
        layer = self.build_layer(name)
        got = layer.get_config()
        expected = {
            "trainable": False,
            "dtype": "float32",
            **self.common_config,
            **self.extra_config_dict[name],
        }
        assert got == expected

    @pytest.mark.parametrize(
        ("name", "moving_param_shape", "fixed_param_shape"),
        [
            ("affine", (4, 3), (4, 3)),
            ("ddf", (*moving_image_size, 3), (*fixed_image_size, 3)),
        ],
    )
    def test_gen_transform_params(self, name, moving_param_shape, fixed_param_shape):
        """
        Check return shapes and moving/fixed params should be different.

        :param name: name of the layer
        """
        layer = self.build_layer(name)
        moving, fixed = layer._gen_transform_params()
        assert moving.shape == (self.batch_size, *moving_param_shape)
        assert fixed.shape == (self.batch_size, *fixed_param_shape)
        assert not is_equal_np(moving, fixed)

    @pytest.mark.parametrize("name", ["affine", "ddf"])
    def test__transform(self, name):
        """
        Check return shapes.

        :param name: name of the layer
        """
        layer = self.build_layer(name)
        moving_image = tf.random.uniform(
            shape=(self.batch_size, *self.moving_image_size)
        )
        moving_params, _ = layer._gen_transform_params()
        transformed = layer._transform(
            image=moving_image,
            grid_ref=layer._moving_grid_ref,
            params=moving_params,
        )
        assert transformed.shape == moving_image.shape

    @pytest.mark.parametrize("name", ["affine", "ddf"])
    @pytest.mark.parametrize("labeled", [True, False])
    def test_call(self, name, labeled):
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
        for k in inputs.keys():
            assert outputs[k].shape == inputs[k].shape
