import pytest

from deepreg.model.network.build import build_model
from deepreg.registry import REGISTRY


class TestBuildModel:
    moving_image_size = (4, 8, 16)
    fixed_image_size = (8, 16, 24)
    index_size = 2
    batch_size = 2
    train_config = {
        "backbone": {
            "name": "local",
            "num_channel_initial": 4,
            "extract_levels": [1, 2, 3],
        },
        "loss": {
            "dissimilarity": {
                "image": {"name": "lncc", "weight": 0.1},
                "label": {
                    "name": "dice",
                    "weight": 1,
                    "scales": [0, 1, 2, 4, 8, 16, 32],
                },
            },
            "regularization": {"weight": 0.0, "name": "bending"},
        },
    }

    @pytest.mark.parametrize(
        "method,backbone",
        [
            ("ddf", "local"),
            ("dvf", "local"),
            ("conditional", "local"),
            ("affine", "global"),
        ],
    )
    def test_build(self, method, backbone):
        train_config = self.train_config.copy()
        train_config["method"] = method
        train_config["backbone"]["name"] = backbone
        build_model(
            moving_image_size=self.moving_image_size,
            fixed_image_size=self.fixed_image_size,
            index_size=self.index_size,
            labeled=True,
            batch_size=self.batch_size,
            train_config=train_config,
            registry=REGISTRY,
        )

    def test_build_err(self):
        train_config = self.train_config.copy()
        train_config["method"] = "unknown"
        with pytest.raises(ValueError) as err_info:
            build_model(
                moving_image_size=self.moving_image_size,
                fixed_image_size=self.fixed_image_size,
                index_size=self.index_size,
                labeled=True,
                batch_size=self.batch_size,
                train_config=train_config,
                registry=REGISTRY,
            )
        assert "Unknown method" in str(err_info.value)
