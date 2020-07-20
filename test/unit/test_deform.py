# coding=utf-8

"""
Tests for deepreg/model/loss/deform.py in
pytest style
"""
from types import FunctionType
import numpy as np
import pytest
import tensorflow as tf
import deepreg.model.loss.deform2 as deform

def assertTensorsEqual(x, y):
    """
    given two tf tensors return True/False (not tf tensor)
    tolerate small errors
    :param x:
    :param y:
    :return:
    """
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    return tf.reduce_max(tf.abs(x - y)).numpy() < 1e-6

def assertTensorsShapeEqual(x, y):
    """assert the tensors have the same shape"""
    return x.shape == y.shape

def test_gradient_dx():
    """test the calculation of gradient of a 3D images along x-axis"""
    tensor = np.ones([4, 50, 50, 50])
    get = deform.gradient_dx(tensor)
    print(get.shape)

def test_gradient_dy():
    """test the calculation of gradient of a 3D images along y-axis"""
    pass

def test_gradient_dz():
    """test the calculation of gradient of a 3D images along z-axis"""
    pass

def test_gradient_txyz():
    """test the calculation of gradient of a 3D images along xyz-axis"""
    pass

def test_compute_gradient_l1norm():
    """test the calculation of l1 norm for image gradients"""
    pass

def test_compute_gradient_else():
    """test the calculation of l2 norm for image gradients"""
    pass

def test_compute_bending_energy():
    """test the calculation of bending energy"""
    pass

test_gradient_dx()