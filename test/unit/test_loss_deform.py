# coding=utf-8

"""
Tests for deepreg/model/loss/deform.py in pytest style
"""
from test.unit.util import is_equal_tf

import pytest
import tensorflow as tf

import deepreg.loss.deform as deform


def test_gradient_dx():
    """test the calculation of gradient of a 3D images along x-axis"""
    tensor = tf.ones([4, 50, 50, 50])
    get = deform.gradient_dx(tensor)
    expect = tf.zeros([4, 48, 48, 48])
    assert is_equal_tf(get, expect)


def test_gradient_dy():
    """test the calculation of gradient of a 3D images along y-axis"""
    tensor = tf.ones([4, 50, 50, 50])
    get = deform.gradient_dy(tensor)
    expect = tf.zeros([4, 48, 48, 48])
    assert is_equal_tf(get, expect)


def test_gradient_dz():
    """test the calculation of gradient of a 3D images along z-axis"""
    tensor = tf.ones([4, 50, 50, 50])
    get = deform.gradient_dz(tensor)
    expect = tf.zeros([4, 48, 48, 48])
    assert is_equal_tf(get, expect)


def test_gradient_dxyz():
    """test the calculation of gradient of a 3D images along xyz-axis"""
    # gradient_dx
    tensor = tf.ones([4, 50, 50, 50, 3])
    get = deform.gradient_dxyz(tensor, deform.gradient_dx)
    expect = tf.zeros([4, 48, 48, 48, 3])
    assert is_equal_tf(get, expect)

    # gradient_dy
    tensor = tf.ones([4, 50, 50, 50, 3])
    get = deform.gradient_dxyz(tensor, deform.gradient_dy)
    expect = tf.zeros([4, 48, 48, 48, 3])
    assert is_equal_tf(get, expect)

    # gradient_dz
    tensor = tf.ones([4, 50, 50, 50, 3])
    get = deform.gradient_dxyz(tensor, deform.gradient_dz)
    expect = tf.zeros([4, 48, 48, 48, 3])
    assert is_equal_tf(get, expect)


class TestGradientNorm:
    @pytest.mark.parametrize("l1", [True, False])
    def test_call(self, l1):
        tensor = tf.ones([4, 50, 50, 50, 3])
        got = deform.GradientNorm(l1=l1)(tensor)
        expected = tf.zeros(
            [
                4,
            ]
        )
        assert is_equal_tf(got, expected)

    def test_get_config(self):
        got = deform.GradientNorm().get_config()
        expected = {
            "name": "GradientNorm",
            "l1": False,
            "dtype": "float32",
            "trainable": True,
        }
        assert got == expected


def test_bending_energy():
    """test the calculation of bending energy"""
    tensor = tf.ones([4, 50, 50, 50, 3])
    got = deform.BendingEnergy()(tensor)
    expected = tf.zeros(
        [
            4,
        ]
    )
    assert is_equal_tf(got, expected)
