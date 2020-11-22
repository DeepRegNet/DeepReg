# coding=utf-8

"""
Tests for deepreg/model/network/util
"""
import pytest

import deepreg.model.network.util as util


class TestBuildBackbone:
    def test_wrong_image_size(self):
        with pytest.raises(ValueError) as err_info:
            util.build_backbone(
                image_size=(1, 1, 1, 1), out_channels=1, config={}, method_name="ddf"
            )
        assert "image_size must be tuple of length 3" in str(err_info.value)

    def test_wrong_method_name(self):
        with pytest.raises(ValueError) as err_info:
            util.build_backbone(
                image_size=(1, 2, 3),
                out_channels=1,
                config={"backbone": "local"},
                method_name="wrong",
            )
        assert (
            "method name has to be one of ddf/dvf/conditional/affine in build_backbone"
            in str(err_info.value)
        )

    @pytest.mark.parametrize("method_name", ["ddf", "dvf", "conditional", "affine"])
    @pytest.mark.parametrize("out_channels", [1, 2, 3])
    @pytest.mark.parametrize("backbone_name", ["local", "global"])
    def test_local_global_backbone(self, method_name, out_channels, backbone_name):
        """Only test the function returns successfully"""
        util.build_backbone(
            image_size=(2, 3, 4),
            out_channels=out_channels,
            config={
                "name": backbone_name,
                "num_channel_initial": 4,
                "extract_levels": [1, 2, 3],
            },
            method_name=method_name,
        )

    @pytest.mark.parametrize("method_name", ["ddf", "dvf", "conditional", "affine"])
    @pytest.mark.parametrize("out_channels", [1, 2, 3])
    def test_unet_backbone(self, method_name, out_channels):
        """Only test the function returns successfully"""
        util.build_backbone(
            image_size=(2, 3, 4),
            out_channels=out_channels,
            config={
                "name": "unet",
                "num_channel_initial": 4,
                "depth": 4,
            },
            method_name=method_name,
        )


class TestBuildInputs:
    moving_image_size = (2, 3, 4)
    fixed_image_size = (1, 2, 3)
    index_size = 3
    batch_size = 2

    @pytest.mark.parametrize("labeled", [True, False])
    def test_input_shape(self, labeled):
        (
            moving_image,
            fixed_image,
            moving_label,
            fixed_label,
            indices,
        ) = util.build_inputs(
            moving_image_size=self.moving_image_size,
            fixed_image_size=self.fixed_image_size,
            index_size=self.index_size,
            batch_size=self.batch_size,
            labeled=labeled,
        )
        assert moving_image.shape == (self.batch_size, *self.moving_image_size)
        assert fixed_image.shape == (self.batch_size, *self.fixed_image_size)
        assert indices.shape == (self.batch_size, self.index_size)
        if labeled:
            assert moving_label.shape == (self.batch_size, *self.moving_image_size)
            assert fixed_label.shape == (self.batch_size, *self.fixed_image_size)
        else:
            assert moving_label is None
            assert fixed_label is None
