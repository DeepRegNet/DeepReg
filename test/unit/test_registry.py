import pytest

from deepreg.registry import BACKBONE_CLASS, LOSS_CLASS, REGISTRY


class TestRegistry:
    @pytest.fixture()
    def reg(self):
        return REGISTRY.copy()

    @pytest.mark.parametrize(
        "category,key,err_msg",
        [
            ("unknown_category", "key", "Unknown category"),
            (BACKBONE_CLASS, "unet", "has been registered"),
        ],
    )
    def test_register_err(self, category, key, err_msg, reg):
        with pytest.raises(ValueError) as err_info:
            reg.register(category=category, name=key, cls=0)
        assert err_msg in str(err_info.value)

    @pytest.mark.parametrize(
        "category,key,force",
        [
            (BACKBONE_CLASS, "unet", True),
            (BACKBONE_CLASS, "vnet", False),
            (LOSS_CLASS, "dice", True),
            (LOSS_CLASS, "Dice", False),
        ],
    )
    def test_register(self, category, key, force, reg):
        value = 0
        reg.register(category=category, name=key, cls=value, force=force)
        assert reg._dict[(category, key)] == value
        assert reg.get(category, key) == value

    @pytest.mark.parametrize(
        "category,key",
        [
            (BACKBONE_CLASS, "unet"),
            (LOSS_CLASS, "dice"),
        ],
    )
    def test_get_backbone(self, category, key, reg):
        # no error means the unet has been registered
        _ = reg.get(category, key)

    def test_get_err(self, reg):
        with pytest.raises(ValueError) as err_info:
            reg.get(BACKBONE_CLASS, "wrong_key")
        assert "has not been registered" in str(err_info.value)

    @pytest.mark.parametrize(
        "category,config,err_msg",
        [
            (BACKBONE_CLASS, [], "config must be a dict"),
            (BACKBONE_CLASS, {}, "`config` must contain the key `name`"),
            (
                BACKBONE_CLASS,
                {"name": "unet"},
                "Configuration is not compatible for Class",
            ),
        ],
    )
    def test_build_from_config_err(self, category, config, err_msg, reg):
        with pytest.raises(ValueError) as err_info:
            reg.build_from_config(category=category, config=config)
        assert err_msg in str(err_info.value)

    @pytest.mark.parametrize(
        "category,config",
        [
            (
                BACKBONE_CLASS,
                dict(
                    name="unet",
                    image_size=[1, 2, 3],
                    out_channels=3,
                    num_channel_initial=3,
                    depth=5,
                    out_kernel_initializer="he_normal",
                    out_activation="softmax",
                ),
            ),
            (LOSS_CLASS, dict(name="dice")),
        ],
    )
    def test_build_from_config(self, category, config, reg):
        _ = reg.build_from_config(category=category, config=config)
