# coding=utf-8

"""
Tests for deepreg/model/network/ddf_dvf.py
"""
import tensorflow as tf

from deepreg.model.network.ddf_dvf import ddf_dvf_forward
from deepreg.model.network.util import build_backbone


def test_ddf_dvf_forward():
    """
    Testing that ddf_dvf_forward function returns the tensors with correct shapes
    """

    moving_image_size = (1, 3, 5)
    fixed_image_size = (2, 4, 6)
    batch_size = 1

    local_net = build_backbone(
        image_size=fixed_image_size,
        out_channels=3,
        model_config={
            "backbone": "local",
            "local": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
        },
        method_name="ddf",
    )

    # ddf mode
    dvf, ddf, pred_fixed_image, pred_fixed_label, grid_fixed = ddf_dvf_forward(
        backbone=local_net,
        moving_image=tf.random.uniform((batch_size,) + moving_image_size),
        fixed_image=tf.random.uniform((batch_size,) + fixed_image_size),
        moving_label=tf.random.uniform((batch_size,) + moving_image_size),
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        output_dvf=False,
    )
    assert dvf is None
    assert ddf.shape == (batch_size,) + fixed_image_size + (3,)
    assert pred_fixed_image.shape == (batch_size,) + fixed_image_size
    assert pred_fixed_label.shape == (batch_size,) + fixed_image_size
    assert grid_fixed.shape == fixed_image_size + (3,)

    # dvf mode
    dvf, ddf, pred_fixed_image, pred_fixed_label, grid_fixed = ddf_dvf_forward(
        backbone=local_net,
        moving_image=tf.random.uniform((batch_size,) + moving_image_size),
        fixed_image=tf.random.uniform((batch_size,) + fixed_image_size),
        moving_label=tf.random.uniform((batch_size,) + moving_image_size),
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        output_dvf=True,
    )
    assert dvf.shape == (batch_size,) + fixed_image_size + (3,)
    assert ddf.shape == (batch_size,) + fixed_image_size + (3,)
    assert pred_fixed_image.shape == (batch_size,) + fixed_image_size
    assert pred_fixed_label.shape == (batch_size,) + fixed_image_size
    assert grid_fixed.shape == fixed_image_size + (3,)
