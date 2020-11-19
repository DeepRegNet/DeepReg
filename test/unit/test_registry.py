import pytest

from deepreg.registry import Registry


class TestRegistry:
    @pytest.mark.parametrize(
        "category,key,err_msg",
        [
            ("unknown_category", "key", "Unknown category"),
            ("backbone_class", "unet", "has been registered"),
        ],
    )
    def test_register_err(self, category, key, err_msg):
        registry = Registry()
        with pytest.raises(ValueError) as err_info:
            registry.register(category, key, 0)
        assert err_msg in str(err_info.value)

    def test_register(self):
        category, key, value = "backbone_class", "test_key", 0
        registry = Registry()
        registry.register(category, key, value)
        assert registry._dict[(category, key)] == value

    def test_get(self):
        category, key, value = "backbone_class", "test_key", 0
        registry = Registry()
        registry.register(category, key, value)
        assert registry.get(category, key) == value

    def test_get_err(self):
        registry = Registry()
        with pytest.raises(ValueError) as err_info:
            registry.get("backbone_class", "wrong_key")
        assert "has not been registered" in str(err_info.value)
