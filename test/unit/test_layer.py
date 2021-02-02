# coding=utf-8

"""
Tests for deepreg/model/layer
"""

import numpy as np
import pytest
import tensorflow as tf

import deepreg.model.layer as layer


@pytest.mark.parametrize("layer_name", ["conv3d", "deconv3d"])
@pytest.mark.parametrize("norm_name", ["batch", "layer"])
@pytest.mark.parametrize("activation", ["relu", "elu"])
def test_norm_block(layer_name: str, norm_name: str, activation: str):
    """
    Test output shapes and configs.

    :param layer_name: layer_name for layer definition
    :param norm_name: norm_name for layer definition
    :param activation: activation for layer definition
    """
    input_size = (2, 3, 4, 5, 6)  # (batch, *shape, ch)
    norm_block = layer.NormBlock(
        layer_name=layer_name,
        norm_name=norm_name,
        activation=activation,
        filters=3,
        kernel_size=1,
        padding="same",
    )
    inputs = tf.ones(shape=input_size)
    outputs = norm_block(inputs)
    assert outputs.shape == input_size[:-1] + (3,)

    config = norm_block.get_config()
    assert config == dict(
        layer_name=layer_name,
        norm_name=norm_name,
        activation=activation,
        filters=3,
        kernel_size=1,
        padding="same",
        name="norm_block",
        trainable=True,
        dtype="float32",
    )


class TestWarping:
    @pytest.mark.parametrize(
        ("moving_image_size", "fixed_image_size"),
        [
            ((1, 2, 3), (3, 4, 5)),
            ((1, 2, 3), (1, 2, 3)),
        ],
    )
    def test_forward(self, moving_image_size, fixed_image_size):
        batch_size = 2
        image = tf.ones(shape=(batch_size,) + moving_image_size)
        ddf = tf.ones(shape=(batch_size,) + fixed_image_size + (3,))
        outputs = layer.Warping(fixed_image_size=fixed_image_size)([ddf, image])
        assert outputs.shape == (batch_size, *fixed_image_size)

    def test_get_config(self):
        warping = layer.Warping(fixed_image_size=(2, 3, 4))
        config = warping.get_config()
        assert config == dict(
            fixed_image_size=(2, 3, 4),
            name="warping",
            trainable=True,
            dtype="float32",
        )


@pytest.mark.parametrize("layer_name", ["conv3d", "deconv3d"])
@pytest.mark.parametrize("norm_name", ["batch", "layer"])
@pytest.mark.parametrize("activation", ["relu", "elu"])
@pytest.mark.parametrize("num_layers", [2, 3])
def test_res_block(layer_name: str, norm_name: str, activation: str, num_layers: int):
    """
    Test output shapes and configs.

    :param layer_name: layer_name for layer definition
    :param norm_name: norm_name for layer definition
    :param activation: activation for layer definition
    :param num_layers: number of blocks in res block
    """
    ch = 3
    input_size = (2, 3, 4, 5, ch)  # (batch, *shape, ch)
    res_block = layer.ResidualBlock(
        layer_name=layer_name,
        num_layers=num_layers,
        norm_name=norm_name,
        activation=activation,
        filters=ch,
        kernel_size=3,
        padding="same",
    )
    inputs = tf.ones(shape=input_size)
    outputs = res_block(inputs)
    assert outputs.shape == input_size[:-1] + (3,)

    config = res_block.get_config()
    assert config == dict(
        layer_name=layer_name,
        num_layers=num_layers,
        norm_name=norm_name,
        activation=activation,
        filters=ch,
        kernel_size=3,
        padding="same",
        name="res_block",
        trainable=True,
        dtype="float32",
    )


class TestIntDVF:
    def test_forward(self):
        """
        Test output shape and config.
        """

        fixed_image_size = (8, 9, 10)
        input_shape = (2, *fixed_image_size, 3)

        int_layer = layer.IntDVF(fixed_image_size=fixed_image_size)

        inputs = tf.ones(shape=input_shape)
        outputs = int_layer(inputs)
        assert outputs.shape == input_shape

        config = int_layer.get_config()
        assert config == dict(
            fixed_image_size=fixed_image_size,
            num_steps=7,
            name="int_dvf",
            trainable=True,
            dtype="float32",
        )

    def test_err(self):
        with pytest.raises(AssertionError):
            layer.IntDVF(fixed_image_size=(2, 3))


class TestResizeCPTransform:
    @pytest.mark.parametrize(
        "parameter,cp_spacing", [((8, 8, 8), 8), ((8, 24, 16), (8, 24, 16))]
    )
    def test_attributes(self, parameter, cp_spacing):
        model = layer.ResizeCPTransform(cp_spacing)

        if isinstance(cp_spacing, int):
            cp_spacing = [cp_spacing] * 3
        assert list(model.cp_spacing) == list(parameter)
        assert model.kernel_sigma == [0.44 * cp for cp in cp_spacing]

    @pytest.mark.parametrize(
        "input_size,output_size,cp_spacing",
        [
            ((1, 8, 8, 8, 3), (12, 8, 12), (8, 16, 8)),
            ((1, 8, 8, 8, 3), (12, 12, 12), 8),
        ],
    )
    def test_build(self, input_size, output_size, cp_spacing):
        model = layer.ResizeCPTransform(cp_spacing)
        model.build(input_size)

        assert [a == b for a, b, in zip(model._output_shape, output_size)]

    @pytest.mark.parametrize(
        "input_size,output_size,cp_spacing",
        [
            ((1, 68, 68, 68, 3), (1, 12, 8, 12, 3), (8, 16, 8)),
            ((1, 68, 68, 68, 3), (1, 12, 12, 12, 3), 8),
        ],
    )
    def test_call(self, input_size, output_size, cp_spacing):
        model = layer.ResizeCPTransform(cp_spacing)
        model.build(input_size)

        input = tf.random.normal(shape=input_size, dtype=tf.float32)
        output = model(input)

        assert output.shape == output_size


class TestBSplines3DTransform:
    """
    Test the layer.BSplines3DTransform class,
    its default attributes and its call() function.
    """

    @pytest.mark.parametrize(
        "input_size,cp",
        [((1, 8, 8, 8, 3), 8), ((1, 8, 8, 8, 3), (8, 16, 12))],
    )
    def test_init(self, input_size, cp):
        model = layer.BSplines3DTransform(cp, input_size[1:-1])

        if isinstance(cp, int):
            cp = (cp, cp, cp)

        assert model.cp_spacing == cp

    @pytest.mark.parametrize(
        "input_size,cp",
        [((1, 8, 8, 8, 3), (8, 8, 8)), ((1, 8, 8, 8, 3), (8, 16, 12))],
    )
    def generate_filter_coefficients(self, cp_spacing):

        b = {
            0: lambda u: np.float64((1 - u) ** 3 / 6),
            1: lambda u: np.float64((3 * (u ** 3) - 6 * (u ** 2) + 4) / 6),
            2: lambda u: np.float64((-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6),
            3: lambda u: np.float64(u ** 3 / 6),
        }

        filters = np.zeros(
            (
                4 * cp_spacing[0],
                4 * cp_spacing[1],
                4 * cp_spacing[2],
                3,
                3,
            ),
            dtype=np.float32,
        )

        for u in range(cp_spacing[0]):
            for v in range(cp_spacing[1]):
                for w in range(cp_spacing[2]):
                    for x in range(4):
                        for y in range(4):
                            for z in range(4):
                                for it_dim in range(3):
                                    u_norm = 1 - (u + 0.5) / cp_spacing[0]
                                    v_norm = 1 - (v + 0.5) / cp_spacing[1]
                                    w_norm = 1 - (w + 0.5) / cp_spacing[2]
                                    filters[
                                        x * cp_spacing[0] + u,
                                        y * cp_spacing[1] + v,
                                        z * cp_spacing[2] + w,
                                        it_dim,
                                        it_dim,
                                    ] = (
                                        b[x](u_norm) * b[y](v_norm) * b[z](w_norm)
                                    )
        return filters

    @pytest.mark.parametrize(
        "input_size,cp",
        [((1, 8, 8, 8, 3), (8, 8, 8)), ((1, 8, 8, 8, 3), (8, 16, 12))],
    )
    def test_build(self, input_size, cp):
        model = layer.BSplines3DTransform(cp, input_size[1:-1])

        model.build(input_size)
        assert model.filter.shape == (
            4 * cp[0],
            4 * cp[1],
            4 * cp[2],
            3,
            3,
        )

    @pytest.mark.parametrize(
        "input_size,cp",
        [((1, 8, 8, 8, 3), (8, 8, 8)), ((1, 8, 8, 8, 3), (8, 16, 12))],
    )
    def test_coefficients(self, input_size, cp):

        filters = self.generate_filter_coefficients(cp_spacing=cp)

        model = layer.BSplines3DTransform(cp, input_size[1:-1])
        model.build(input_size)

        assert np.allclose(filters, model.filter.numpy(), atol=1e-8)

    @pytest.mark.parametrize(
        "input_size,cp",
        [((1, 8, 8, 8, 3), (8, 8, 8)), ((1, 8, 8, 8, 3), (8, 16, 12))],
    )
    def test_interpolation(self, input_size, cp):
        model = layer.BSplines3DTransform(cp, input_size[1:-1])
        model.build(input_size)

        vol_shape = input_size[1:-1]
        num_cp = (
            [input_size[0]]
            + [int(np.ceil(isize / cpsize) + 3) for isize, cpsize in zip(vol_shape, cp)]
            + [input_size[-1]]
        )

        field = tf.random.normal(shape=num_cp, dtype=tf.float32)

        ddf = model.call(field)
        assert ddf.shape == input_size


class TestResize3d:
    @pytest.mark.parametrize(
        ("input_shape", "resize_shape", "output_shape"),
        [
            ((1, 2, 3), (3, 4, 5), (3, 4, 5)),
            ((2, 1, 2, 3), (3, 4, 5), (2, 3, 4, 5)),
            ((2, 1, 2, 3, 1), (3, 4, 5), (2, 3, 4, 5, 1)),
            ((2, 1, 2, 3, 6), (3, 4, 5), (2, 3, 4, 5, 6)),
            ((1, 2, 3), (1, 2, 3), (1, 2, 3)),
            ((2, 1, 2, 3), (1, 2, 3), (2, 1, 2, 3)),
            ((2, 1, 2, 3, 1), (1, 2, 3), (2, 1, 2, 3, 1)),
            ((2, 1, 2, 3, 6), (1, 2, 3), (2, 1, 2, 3, 6)),
        ],
    )
    def test_forward(self, input_shape, resize_shape, output_shape):
        inputs = tf.ones(shape=input_shape)
        outputs = layer.Resize3d(shape=resize_shape)(inputs)
        assert outputs.shape == output_shape

    def test_get_config(self):
        resize = layer.Resize3d(shape=(2, 3, 4))
        config = resize.get_config()
        assert config == dict(
            shape=(2, 3, 4),
            method=tf.image.ResizeMethod.BILINEAR,
            name="resize3d",
            trainable=True,
            dtype="float32",
        )

    def test_shape_err(self):
        with pytest.raises(AssertionError):
            layer.Resize3d(shape=(2, 3))

    def test_image_shape_err(self):
        with pytest.raises(ValueError) as err_info:
            resize = layer.Resize3d(shape=(2, 3, 4))
            resize(tf.ones(1, 1))
        assert "Resize3d takes input image of dimension 3 or 4 or 5" in str(
            err_info.value
        )
