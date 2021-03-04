# coding=utf-8

"""
Tests for deepreg/model/backbone/interface.py
"""
import deepreg.model.backbone as backbone


def test_backbone_interface():
    """Test the get_config of the interface"""
    config = dict(
        image_size=(5, 5, 5),
        out_channels=3,
        num_channel_initial=4,
        out_kernel_initializer="zeros",
        out_activation="relu",
        name="test",
    )
    model = backbone.Backbone(**config)
    got = model.get_config()
    assert got == config
