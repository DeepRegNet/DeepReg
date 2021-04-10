# coding=utf-8

"""
Tests for deepreg/_model/network/ddf_dvf.py
"""
import itertools
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from deepreg.model.network import RegistrationModel
from deepreg.registry import REGISTRY

moving_image_size = (1, 3, 5)
fixed_image_size = (2, 4, 6)
index_size = 2
batch_size = 3
backbone_args = {
    "local": {"extract_levels": [1, 2]},
    "global": {"extract_levels": [1, 2]},
    "unet": {"depth": 2},
}
config = {
    "backbone": {"num_channel_initial": 4, "control_points": 2},
    "loss": {
        "image": {"name": "lncc", "weight": 0.1},
        "label": {
            "name": "dice",
            "weight": 1,
            "scales": [0, 1],
        },
        "regularization": {"weight": 0.1, "name": "bending"},
    },
}

config_multiple_losses = {
    "backbone": {"num_channel_initial": 4, "control_points": 2},
    "loss": {
        "image": [
            {"name": "lncc", "weight": 0.1},
            {"name": "ssd", "weight": 0.1},
            {"name": "gmi", "weight": 0.1},
        ],
        "label": {
            "name": "dice",
            "weight": 1,
            "scales": [0, 1],
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
    copied = deepcopy(config)
    copied["method"] = method
    copied["backbone"]["name"] = backbone  # type: ignore
    if method == "conditional":
        copied["backbone"].pop("control_points", None)  # type: ignore
    copied["backbone"].update(backbone_args[backbone])  # type: ignore
    return REGISTRY.build_model(  # type: ignore
        config=dict(
            name=method,  # TODO we store method twice
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            index_size=index_size,
            labeled=labeled,
            batch_size=batch_size,
            config=copied,
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

    def test_get_config(self, empty_model, labeled):
        got = empty_model.get_config()
        expected = dict(
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            index_size=index_size,
            labeled=labeled,
            batch_size=batch_size,
            config=dict(),
            name="RegistrationModel",
        )
        assert got == expected

    def test_build_inputs(self, empty_model, labeled):
        inputs = empty_model.build_inputs()
        expected_inputs_len = 5 if labeled else 3
        assert len(inputs) == expected_inputs_len

        moving_image = inputs["moving_image"]
        fixed_image = inputs["fixed_image"]
        indices = inputs["indices"]
        assert moving_image.shape == (batch_size, *moving_image_size)
        assert fixed_image.shape == (batch_size, *fixed_image_size)
        assert indices.shape == (batch_size, index_size)

        if labeled:
            moving_label = inputs["moving_label"]
            fixed_label = inputs["fixed_label"]
            assert moving_label.shape == (batch_size, *moving_image_size)
            assert fixed_label.shape == (batch_size, *fixed_image_size)

    def test_concat_images(self, empty_model, labeled):
        inputs = empty_model.build_inputs()
        moving_image = inputs["moving_image"]
        fixed_image = inputs["fixed_image"]
        if labeled:
            moving_label = inputs["moving_label"]
            images = empty_model.concat_images(moving_image, fixed_image, moving_label)
            assert images.shape == (batch_size, *fixed_image_size, 3)
        else:
            images = empty_model.concat_images(moving_image, fixed_image)
            assert images.shape == (batch_size, *fixed_image_size, 2)


class TestBuildLoss:
    params = [
        dict(config=config, option=0, expected=2),
        dict(config=config, option=1, expected=2),
        dict(config=config, option=2, expected=3),
        dict(config=config_multiple_losses, option=3, expected=5),
    ]

    def test_image_loss(self, config: dict, option: int, expected: int):
        method = "ddf"
        backbone = "local"
        labeled = True
        copied = deepcopy(config)
        copied["method"] = method
        copied["backbone"]["name"] = backbone
        copied["backbone"] = {
            **backbone_args[backbone],  # type: ignore
            **copied["backbone"],
        }

        if option == 0:
            # remove image loss config, so loss is not used
            copied["loss"].pop("image")
        elif option == 1:
            # set image loss weight to zero, so loss is not used
            copied["loss"]["image"]["weight"] = 0.0
        elif option == 2:
            # remove image loss weight, so loss is used with default weight 1
            copied["loss"]["image"].pop("weight")

        ddf_model = REGISTRY.build_model(
            config=dict(
                name=method,  # TODO we store method twice
                moving_image_size=moving_image_size,
                fixed_image_size=fixed_image_size,
                index_size=index_size,
                labeled=labeled,
                batch_size=batch_size,
                config=copied,
            )
        )

        assert len(ddf_model._model.losses) == expected  # type: ignore


class TestDDFModel:
    params = [
        dict(method=method, labeled=labeled, backbone=backbone)
        for method, labeled, backbone in itertools.product(
            ["ddf"], [True, False], ["local", "global", "unet"]
        )
    ]

    def test_build_model(self, model, labeled, backbone):
        expected_outputs_len = 3 if labeled else 2
        if backbone == "global":
            expected_outputs_len += 1
            theta = model._outputs["theta"]
            assert theta.shape == (batch_size, 4, 3)
        assert len(model._outputs) == expected_outputs_len

        ddf = model._outputs["ddf"]
        pred_fixed_image = model._outputs["pred_fixed_image"]
        assert ddf.shape == (batch_size, *fixed_image_size, 3)
        assert pred_fixed_image.shape == (batch_size, *fixed_image_size)

        if labeled:
            pred_fixed_label = model._outputs["pred_fixed_label"]
            assert pred_fixed_label.shape == (batch_size, *fixed_image_size)

    def test_build_loss(self, model, labeled, backbone):
        expected = 3 if labeled else 2
        assert len(model._model.losses) == expected

    def test_postprocess(self, model, labeled, backbone):
        indices, processed = model.postprocess(
            inputs=model._inputs, outputs=model._outputs
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
        expected_outputs_len = 4 if labeled else 3
        assert len(model._outputs) == expected_outputs_len

        dvf = model._outputs["dvf"]
        ddf = model._outputs["ddf"]
        pred_fixed_image = model._outputs["pred_fixed_image"]
        assert dvf.shape == (batch_size, *fixed_image_size, 3)
        assert ddf.shape == (batch_size, *fixed_image_size, 3)
        assert pred_fixed_image.shape == (batch_size, *fixed_image_size)

        if labeled:
            pred_fixed_label = model._outputs["pred_fixed_label"]
            assert pred_fixed_label.shape == (batch_size, *fixed_image_size)

    def test_build_loss(self, model, labeled, backbone):
        expected = 3 if labeled else 2
        assert len(model._model.losses) == expected

    def test_postprocess(self, model, labeled, backbone):
        indices, processed = model.postprocess(
            inputs=model._inputs, outputs=model._outputs
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
        assert len(model._outputs) == 1
        pred_fixed_label = model._outputs["pred_fixed_label"]
        assert pred_fixed_label.shape == (batch_size, *fixed_image_size)

    def test_build_loss(self, model, labeled, backbone):
        assert len(model._model.losses) == 1

    def test_postprocess(self, model, labeled, backbone):
        indices, processed = model.postprocess(
            inputs=model._inputs, outputs=model._outputs
        )
        assert indices.shape == (batch_size, index_size)
        assert len(processed) == 5
