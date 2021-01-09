# coding=utf-8

"""
Tests for deepreg/_model/network/ddf_dvf.py
"""
import itertools
from unittest.mock import MagicMock, patch

import pytest

from deepreg.model.network.ddf_dvf import RegistrationModel
from deepreg.registry import REGISTRY

moving_image_size = (1, 3, 5)
fixed_image_size = (2, 4, 6)
index_size = 2
batch_size = 3
backbone_args = {
    "local": {"extract_levels": [1, 2, 3]},
    "global": {"extract_levels": [1, 2, 3]},
    "unet": {"depth": 2},
}
config = {
    "backbone": {
        "num_channel_initial": 4,
    },
    "loss": {
        "image": {"name": "lncc", "weight": 0.1},
        "label": {
            "name": "dice",
            "weight": 1,
            "scales": [0, 1, 2, 4, 8, 16, 32],
        },
        "regularization": {"weight": 0.1, "name": "bending"},
    },
}


@pytest.fixture
def model(method: str, labeled: bool, backbone: str) -> RegistrationModel:
    """
    A specific registration model object.

    :param method: name of method
    :param labeled: whether the data is labeled
    :param backbone: name of backbone
    :return: the built object
    """
    _config = config.copy()
    _config["method"] = method
    _config["backbone"]["name"] = backbone
    _config["backbone"] = {**backbone_args[backbone], **_config["backbone"]}
    return REGISTRY.build_model(
        config=dict(
            name=method,  # TODO we store method twice
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            index_size=index_size,
            labeled=labeled,
            batch_size=batch_size,
            config=_config,
        )
    )


def pytest_generate_tests(metafunc):
    """
    Test parameter generator.

    This function is called once per each test function.
    It takes the attribute `params` from the test class,
    and then use the same `params` for all tests inside the class.
    This is specific for test of registration models only.

    This is modified from the pytest documentation,
    where their version defined the params for each test function separately.

    https://docs.pytest.org/en/stable/example/parametrize.html#parametrizing-test-methods-through-per-class-configuration

    :param metafunc:
    :return:
    """
    #
    funcarglist = metafunc.cls.params
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )


class TestRegistrationModel:
    params = [dict(labeled=True), dict(labeled=False)]

    @pytest.fixture
    def empty_model(self, labeled: bool) -> RegistrationModel:
        """
        A RegistrationModel with build_model and build_loss mocked/overwritten.

        :param labeled: whether the data is labeled
        :return: the mocked object
        """
        with patch.multiple(
            RegistrationModel,
            build_model=MagicMock(return_value=None),
            build_loss=MagicMock(return_value=None),
        ):
            return RegistrationModel(
                moving_image_size=moving_image_size,
                fixed_image_size=fixed_image_size,
                index_size=index_size,
                labeled=labeled,
                batch_size=batch_size,
                config=dict(),
            )

    def test_build_inputs(self, empty_model, labeled):
        inputs = empty_model.build_inputs()
        if labeled:
            assert len(inputs) == 5
            moving_image, fixed_image, indices, moving_label, fixed_label = inputs
            assert moving_label.shape == (batch_size, *moving_image_size)
            assert fixed_label.shape == (batch_size, *fixed_image_size)
        else:
            assert len(inputs) == 3
            moving_image, fixed_image, indices = inputs
        assert moving_image.shape == (batch_size, *moving_image_size)
        assert fixed_image.shape == (batch_size, *fixed_image_size)
        assert indices.shape == (batch_size, index_size)

    def test_concat_images(self, empty_model, labeled):
        inputs = empty_model.build_inputs()
        if labeled:
            moving_image, fixed_image, _, moving_label, _ = inputs
            images = empty_model.concat_images(moving_image, fixed_image, moving_label)
            assert images.shape == (batch_size, *fixed_image_size, 3)
        else:
            moving_image, fixed_image, _ = inputs
            images = empty_model.concat_images(moving_image, fixed_image)
            assert images.shape == (batch_size, *fixed_image_size, 2)


class TestDDFModel:
    params = [
        dict(method=method, labeled=labeled, backbone=backbone)
        for method, labeled, backbone in itertools.product(
            ["ddf"], [True, False], ["local", "global", "unet"]
        )
    ]

    def test_build_model(self, model, labeled, backbone):
        if labeled:
            assert len(model._model.outputs) == 3
            ddf, pred_fixed_image, pred_fixed_label = model._model.outputs
            assert pred_fixed_label.shape == (batch_size, *fixed_image_size)
        else:
            assert len(model._model.outputs) == 2
            ddf, pred_fixed_image = model._model.outputs
        assert ddf.shape == (batch_size, *fixed_image_size, 3)
        assert pred_fixed_image.shape == (batch_size, *fixed_image_size)

    def test_build_loss(self, model, labeled, backbone):
        expected = 3 if labeled else 2
        assert len(model._model.losses) == expected

    def test_postprocess(self, model, labeled, backbone):
        indices, processed = model.postprocess(
            inputs=model._model.inputs, outputs=model._model.outputs
        )
        assert indices.shape == (batch_size, index_size)
        expected = 7 if labeled else 4
        if backbone == "global":
            expected += 1
        assert len(processed) == expected


class TestDVFModel:
    params = [
        dict(method=method, labeled=labeled, backbone=backbone)
        for method, labeled, backbone in itertools.product(
            ["dvf"], [True, False], ["local", "unet"]
        )
    ]

    def test_build_model(self, model, labeled, backbone):
        if labeled:
            assert len(model._model.outputs) == 4
            dvf, ddf, pred_fixed_image, pred_fixed_label = model._model.outputs
            assert pred_fixed_label.shape == (batch_size, *fixed_image_size)
        else:
            assert len(model._model.outputs) == 3
            dvf, ddf, pred_fixed_image = model._model.outputs
        assert dvf.shape == (batch_size, *fixed_image_size, 3)
        assert ddf.shape == (batch_size, *fixed_image_size, 3)
        assert pred_fixed_image.shape == (batch_size, *fixed_image_size)

    def test_build_loss(self, model, labeled, backbone):
        expected = 3 if labeled else 2
        assert len(model._model.losses) == expected

    def test_postprocess(self, model, labeled, backbone):
        indices, processed = model.postprocess(
            inputs=model._model.inputs, outputs=model._model.outputs
        )
        assert indices.shape == (batch_size, index_size)
        expected = 8 if labeled else 5
        assert len(processed) == expected


class TestConditionalModel:
    params = [
        dict(method=method, labeled=labeled, backbone=backbone)
        for method, labeled, backbone in itertools.product(
            ["conditional"], [True], ["local", "unet"]
        )
    ]

    def test_build_model(self, model, labeled, backbone):
        assert len(model._model.outputs) == 1
        pred_fixed_label = model._model.outputs[0]
        assert pred_fixed_label.shape == (batch_size, *fixed_image_size)

    def test_build_loss(self, model, labeled, backbone):
        assert len(model._model.losses) == 1

    def test_postprocess(self, model, labeled, backbone):
        indices, processed = model.postprocess(
            inputs=model._model.inputs, outputs=model._model.outputs
        )
        assert indices.shape == (batch_size, index_size)
        assert len(processed) == 5
