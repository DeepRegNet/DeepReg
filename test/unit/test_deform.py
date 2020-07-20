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
    assert x.shape == y.shape
    return tf.reduce_max(tf.abs(x - y)).numpy() < 1e-6

def test_gradient_dx():
    """test the calculation of gradient of a 3D images along x-axis"""
    tensor = tf.ones([4, 50, 50, 50])
    get = deform.gradient_dx(tensor)
    expect = tf.zeros([4, 48, 48, 48])
    assert assertTensorsEqual(get, expect)

def test_gradient_dy():
    """test the calculation of gradient of a 3D images along y-axis"""
    tensor = tf.ones([4, 50, 50, 50])
    get = deform.gradient_dy(tensor)
    expect = tf.zeros([4, 48, 48, 48])
    assert assertTensorsEqual(get, expect)

def test_gradient_dz():
    """test the calculation of gradient of a 3D images along z-axis"""
    tensor = tf.ones([4, 50, 50, 50])
    get = deform.gradient_dz(tensor)
    expect = tf.zeros([4, 48, 48, 48])
    assert assertTensorsEqual(get, expect)

def test_gradient_txyz():
    """test the calculation of gradient of a 3D images along xyz-axis"""
    tensor = tf.ones([4, 50, 50, 50, 3])
    get = deform.gradient_dz(tensor)
    expect = tf.zeros([4, 48, 48, 48, 3])
    assert assertTensorsEqual(get, expect)

def test_compute_gradient_l1norm():
    """test the calculation of l1 norm for image gradients"""
    tensor = tf.ones([4, 50, 50, 50, 3])
    get = deform.compute_gradient_norm(tensor, l1=True)
    expect = tf.zeros([4])
    assert assertTensorsEqual(get, expect)
    
def test_compute_gradient_else():
    """test the calculation of l2 norm for image gradients"""
    tensor = tf.ones([4, 50, 50, 50, 3])
    get = deform.compute_gradient_norm(tensor)
    expect = tf.zeros([4])
    assert assertTensorsEqual(get, expect)

def test_compute_bending_energy():
    """test the calculation of bending energy"""
    tensor = tf.ones([4, 50, 50, 50, 3])
    get = deform.compute_bending_energy(tensor)
    expect = tf.zeros([4])
    assert assertTensorsEqual(get, expect)

def test_local_displacement_energy_bending():
    """compute bending energy for ddf"""
    tensor = tf.ones([4, 50, 50, 50, 3])
    get = deform.local_displacement_energy(tensor, 'bending')
    expect = tf.zeros([4])
    assert assertTensorsEqual(get, expect)

def test_local_displacement_energy_l1norm():
    """compute l1 norm for ddf"""
    tensor = tf.ones([4, 50, 50, 50, 3])
    get = deform.local_displacement_energy(tensor, 'gradient-l1')
    expect = tf.zeros([4])
    assert assertTensorsEqual(get, expect)

def test_local_displacement_energy_l2norm():
    """compute l2 norm for ddf"""
    tensor = tf.ones([4, 50, 50, 50, 3])
    get = deform.local_displacement_energy(tensor, 'gradient-l2')
    expect = tf.zeros([4])
    assert assertTensorsEqual(get, expect)

