import pytest

from deepreg.registry import REGISTRY


class TestRegistry:
    @pytest.mark.parametrize(
        "category,key,err_msg",
        [
            ("unknown_category", "key", "Unknown category"),
            ("backbone_class", "unet", "has been registered"),
        ],
    )
    def test_register_err(self, category, key, err_msg):
        with pytest.raises(ValueError) as err_info:
            REGISTRY.register(category=category, name=key, cls=0)
        assert err_msg in str(err_info.value)

    def test_register(self):
        category, key, value = "backbone_class", "test_key", 0
        REGISTRY.register(category=category, name=key, cls=value)
        assert REGISTRY._dict[(category, key)] == value
        assert REGISTRY.get(category, key) == value

    def test_get_err(self):
        with pytest.raises(ValueError) as err_info:
            REGISTRY.get("backbone_class", "wrong_key")
        assert "has not been registered" in str(err_info.value)

    def test_backbone(self):
        key = "new_backbone"
        value = 0
        REGISTRY.register_backbone(name=key, cls=value)
        got = REGISTRY.get_backbone(key)
        assert got == value
