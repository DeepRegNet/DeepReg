from typing import Dict

import pytest
import yaml

from deepreg.config.v011 import (
    parse_data,
    parse_image_loss,
    parse_label_loss,
    parse_loss,
    parse_model,
    parse_optimizer,
    parse_reg_loss,
    parse_v011,
)


@pytest.mark.parametrize(
    ("old_config_path", "latest_config_path"),
    [
        (
            "config/test/grouped_mr_heart_v011.yaml",
            "demos/grouped_mr_heart/grouped_mr_heart.yaml",
        ),
        (
            "demos/grouped_mr_heart/grouped_mr_heart.yaml",
            "demos/grouped_mr_heart/grouped_mr_heart.yaml",
        ),
    ],
)
def test_grouped_mr_heart(old_config_path: str, latest_config_path: str):
    with open(old_config_path) as file:
        old_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(latest_config_path) as file:
        latest_config = yaml.load(file, Loader=yaml.FullLoader)
    updated_config = parse_v011(old_config=old_config)
    assert updated_config == latest_config


class TestParseData:
    config_v011 = {
        "dir": {
            "train": "dir_train",
            "test": "dir_test",
        },
        "format": "h5",
        "labeled": True,
        "type": "paired",
    }
    config_latest = {
        "train": {
            "dir": "dir_train",
            "format": "h5",
            "labeled": True,
        },
        "test": {
            "dir": "dir_test",
            "format": "h5",
            "labeled": True,
        },
        "type": "paired",
    }

    @pytest.mark.parametrize(
        ("data_config", "expected"),
        [
            (config_v011, config_latest),
            (config_latest, config_latest),
        ],
    )
    def test_parse(self, data_config: Dict, expected: Dict):
        got = parse_data(data_config=data_config)
        assert got == expected


class TestParseModel:
    config_v011 = {
        "model": {
            "method": "dvf",
            "backbone": "global",
            "global": {"num_channel_initial": 32, "extract_levels": [0, 1, 2]},
        }
    }
    config_latest = {
        "method": "dvf",
        "backbone": {"name": "global", "num_channel_initial": 32, "depth": 2},
    }

    @pytest.mark.parametrize(
        ("model_config", "expected"),
        [
            (config_v011, config_latest),
            (config_v011["model"], config_latest),
            (config_latest, config_latest),
        ],
    )
    def test_parse(self, model_config: Dict, expected: Dict):
        got = parse_model(model_config=model_config)
        assert got == expected


class TestParseLoss:
    config_v011 = {
        "dissimilarity": {
            "image": {
                "name": "lncc",
                "weight": 2.0,
                "lncc": {
                    "kernel_size": 9,
                    "kernel_type": "rectangular",
                },
            },
        }
    }
    config_latest = {
        "image": {
            "name": "lncc",
            "weight": 2.0,
            "kernel_size": 9,
            "kernel_type": "rectangular",
        },
    }

    @pytest.mark.parametrize(
        ("loss_config", "expected"),
        [
            (config_v011, config_latest),
            (config_latest, config_latest),
        ],
    )
    def test_parse(self, loss_config: Dict, expected: Dict):
        got = parse_loss(loss_config=loss_config)
        assert got == expected


class TestParseImageLoss:
    config_v011 = {
        "image": {
            "name": "lncc",
            "weight": 2.0,
            "lncc": {
                "kernel_size": 9,
                "kernel_type": "rectangular",
            },
        },
    }
    config_latest = {
        "image": {
            "name": "lncc",
            "weight": 2.0,
            "kernel_size": 9,
            "kernel_type": "rectangular",
        },
    }

    @pytest.mark.parametrize(
        ("loss_config", "expected"),
        [
            (config_v011, config_latest),
            (config_latest, config_latest),
        ],
    )
    def test_parse(self, loss_config: Dict, expected: Dict):
        got = parse_image_loss(loss_config=loss_config)
        assert got == expected

    def test_parse_multiple_loss(self):
        loss_config = {
            "image": [
                {
                    "name": "lncc",
                    "weight": 0.5,
                    "kernel_size": 9,
                    "kernel_type": "rectangular",
                },
                {
                    "name": "ssd",
                    "weight": 0.5,
                },
            ],
        }

        got = parse_image_loss(loss_config=loss_config)
        assert got == loss_config


class TestParseLabelLoss:
    @pytest.mark.parametrize(
        ("name_loss", "expected_config"),
        [
            (
                "multi_scale",
                {
                    "label": {
                        "name": "ssd",
                        "weight": 2.0,
                        "scales": [0, 1],
                    },
                },
            ),
            (
                "single_scale",
                {
                    "label": {
                        "name": "dice",
                        "weight": 1.0,
                    },
                },
            ),
        ],
    )
    def test_parse_outdated_loss(self, name_loss: str, expected_config: Dict):
        outdated_config = {
            "label": {
                "name": name_loss,
                "single_scale": {
                    "loss_type": "dice_generalized",
                },
                "multi_scale": {
                    "loss_type": "mean-squared",
                    "loss_scales": [0, 1],
                },
            },
        }

        if name_loss == "multi_scale":
            outdated_config["label"]["weight"] = 2.0  # type: ignore

        got = parse_label_loss(loss_config=outdated_config)
        assert got == expected_config

    def test_parse_background_weight(self):
        outdated_config = {
            "label": {
                "name": "dice",
                "weight": 1.0,
                "neg_weight": 2.0,
            },
        }
        expected_config = {
            "label": {
                "name": "dice",
                "weight": 1.0,
                "background_weight": 2.0,
            },
        }
        got = parse_label_loss(loss_config=outdated_config)
        assert got == expected_config

    def test_parse_multiple_loss(self):
        loss_config = {
            "label": [
                {
                    "name": "dice",
                    "weight": 1.0,
                },
                {
                    "name": "cross-entropy",
                    "weight": 1.0,
                },
            ],
        }

        got = parse_label_loss(loss_config=loss_config)
        assert got == loss_config


class TestParseRegularizationLoss:
    @pytest.mark.parametrize(
        ("energy_type", "loss_name", "extra_args"),
        [
            ("bending", "bending", {}),
            ("gradient-l2", "gradient", {"l1": False}),
            ("gradient-l1", "gradient", {"l1": True}),
        ],
    )
    def test_parse_outdated_loss(
        self, energy_type: str, loss_name: str, extra_args: Dict
    ):

        loss_config = {
            "regularization": {
                "energy_type": energy_type,
                "weight": 2.0,
            }
        }
        expected = {
            "regularization": {
                "name": loss_name,
                "weight": 2.0,
                **extra_args,
            },
        }
        got = parse_reg_loss(loss_config=loss_config)
        assert got == expected

    def test_parse_multiple_reg_loss(self):
        loss_config = {
            "regularization": [
                {
                    "name": "bending",
                    "weight": 2.0,
                },
                {
                    "name": "gradient",
                    "weight": 2.0,
                    "l1": True,
                },
            ],
        }
        got = parse_reg_loss(loss_config=loss_config)
        assert got == loss_config


class TestParseOptimizer:
    config_v011 = {
        "name": "adam",
        "adam": {
            "learning_rate": 1.0e-4,
        },
        "sgd": {
            "learning_rate": 1.0e-4,
            "momentum": 0.9,
        },
    }
    config_latest = {
        "name": "Adam",
        "learning_rate": 1.0e-4,
    }

    @pytest.mark.parametrize(
        ("opt_config", "expected"),
        [
            (config_v011, config_latest),
            (config_latest, config_latest),
        ],
    )
    def test_parse(self, opt_config: Dict, expected: Dict):
        got = parse_optimizer(opt_config=opt_config)
        assert got == expected
