import pytest
import yaml

from deepreg.config.v011 import parse_loss, parse_model, parse_v011


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


class TestParseModel:
    config_v011 = {
        "model": {
            "method": "dvf",
            "backbone": "global",
            "global": {"num_channel_initial": 32},
        }
    }
    config_latest = {
        "method": "dvf",
        "backbone": {"name": "global", "num_channel_initial": 32},
    }

    @pytest.mark.parametrize(
        ("model_config", "expected"),
        [
            (config_v011, config_latest),
            (config_v011["model"], config_latest),
            (config_latest, config_latest),
        ],
    )
    def test_parse(self, model_config: dict, expected: dict):
        got = parse_model(model_config=model_config)
        assert got == expected


class TestParseLoss:
    config_v011 = {
        "loss": {
            "dissimilarity": {
                "image": {"name": "gmi", "weight": 2.0},
                "label": {"name": "multi_scale", "weight": 2.0},
            }
        }
    }

    def test_image(self):
        loss_config = {
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
        expected = {
            "image": {
                "name": "lncc",
                "weight": 2.0,
                "kernel_size": 9,
                "kernel_type": "rectangular",
            },
        }
        got = parse_loss(loss_config=loss_config)
        assert got == expected

    def test_label_multi_scale(self):
        loss_config = {
            "label": {
                "name": "multi_scale",
                "weight": 2.0,
                "multi_scale": {
                    "loss_type": "mean-squared",
                    "loss_scales": [0, 1],
                },
            },
        }
        expected = {
            "label": {
                "name": "ssd",
                "weight": 2.0,
                "scales": [0, 1],
            },
        }
        got = parse_loss(loss_config=loss_config)
        assert got == expected

    def test_label_single_scale(self):
        loss_config = {
            "dissimilarity": {
                "label": {
                    "name": "single_scale",
                    "single_scale": {
                        "loss_type": "dice_generalized",
                    },
                    "multi_scale": {
                        "loss_type": "mean-squared",
                        "loss_scales": [0, 1],
                    },
                },
            }
        }
        expected = {
            "label": {
                "name": "dice",
                "weight": 1.0,
            },
        }
        got = parse_loss(loss_config=loss_config)
        assert got == expected

    @pytest.mark.parametrize(
        ("energy_type", "loss_name", "extra_args"),
        [
            ("bending", "bending", {}),
            ("gradient-l2", "gradient", {"l1": False}),
            ("gradient-l1", "gradient", {"l1": True}),
        ],
    )
    def test_regularization(self, energy_type: str, loss_name: str, extra_args: dict):
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
        got = parse_loss(loss_config=loss_config)
        assert got == expected
