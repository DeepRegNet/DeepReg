from copy import deepcopy
from typing import Callable

BACKBONE_CLASS = "backbone_class"
LOSS_CLASS = "loss_class"
MODEL_CLASS = "model_class"
DATA_AUGMENTATION_CLASS = "da_class"
KNOWN_CATEGORIES = [BACKBONE_CLASS, LOSS_CLASS, MODEL_CLASS, DATA_AUGMENTATION_CLASS]


class Registry:
    """
    Registry maintains a dictionary which maps (category, key) to value.

    Multiple __init__.py files have been modified so that
    when executing `from deepreg.registry import REGISTRY`
    all classes have been registered.

    References:
    - https://github.com/ray-project/ray/blob/00ef1179c012719a17c147a5c3b36d6bdbe97195/python/ray/tune/registry.py#L108
    - https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/builder.py
    - https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py
    - https://towardsdatascience.com/whats-init-for-me-d70a312da583
    """

    def __init__(self):
        self._dict = {}

    def _register(self, category: str, key: str, value: Callable, force: bool):
        """
        Registers the value with the registry.

        :param category: name of the class category
        :param key: unique identity
        :param value: class to be registered
        :param force: if True, overwrite the existing value
            in case the key has been registered.
        """
        # sanity check
        if category not in KNOWN_CATEGORIES:
            raise ValueError(
                f"Unknown category {category} not among {KNOWN_CATEGORIES}"
            )
        if not force and self.contains(category=category, key=key):
            raise ValueError(
                f"Key {key} in category {category} has been registered"
                f" with {self._dict[(category, key)]}"
            )
        # register value
        self._dict[(category, key)] = value

    def contains(self, category: str, key: str):
        return (category, key) in self._dict

    def get(self, category, key):
        if self.contains(category=category, key=key):
            return self._dict[(category, key)]
        raise ValueError(f"Key {key} in category {category} has not been registered.")

    def register(
        self, category: str, name: str, cls: Callable = None, force: bool = False
    ) -> Callable:
        """
        Register a py class.
        A record will be added to `self._dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        :param category: The type of the category.
        :param name: The class name to be registered.
            If not specified, the class name will be used.
        :param force: Whether to override an existing class with the same name.
        :param cls: Class to be registered.
        :return: The given class or a decorator.
        """
        # use it as a normal method: x.register_module(module=SomeClass)
        if cls is not None:
            self._register(category=category, key=name, value=cls, force=force)
            return cls

        # use it as a decorator: @x.register(name, category)
        def decorator(c: Callable) -> Callable:
            self._register(category=category, key=name, value=c, force=force)
            return c

        return decorator

    def build_from_config(
        self, category: str, config: dict, default_args=None
    ) -> object:
        """
        Build a class instance from config dict.

        :param category:
        :param config: Config dict. It should at least contain the key "name".
        :param default_args:
        :return: The constructed object/instance.
        """
        if not isinstance(config, dict):
            raise ValueError(f"config must be a dict, but got {type(config)}")
        if "name" not in config:
            raise ValueError(f"`config` must contain the key `name`, but got {config}")
        args = deepcopy(config)

        # insert key, value pairs if key is not in args
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)

        name = args.pop("name")
        cls = self.get(category=category, key=name)
        try:
            return cls(**args)
        except TypeError as err:
            raise ValueError(
                f"Configuration is not compatible "
                f"for Class {cls} of category {category}.\n"
                f"Potentially an outdated configuration has been used.\n"
                f"Please check the latest documentation of the class"
                f"and arrange the required keys at the same level"
                f" as `name` in configuration file.\n"
                f"{err}"
            )

    def copy(self):
        copied = Registry()
        copied._dict = deepcopy(self._dict)
        return copied

    def register_model(self, name: str, cls: Callable = None, force: bool = False):
        return self.register(category=MODEL_CLASS, name=name, cls=cls, force=force)

    def build_model(self, config: dict, default_args=None):
        return self.build_from_config(
            category=MODEL_CLASS, config=config, default_args=default_args
        )

    def register_backbone(self, name: str, cls: Callable = None, force: bool = False):
        return self.register(category=BACKBONE_CLASS, name=name, cls=cls, force=force)

    def build_backbone(self, config: dict, default_args=None):
        return self.build_from_config(
            category=BACKBONE_CLASS, config=config, default_args=default_args
        )

    def register_loss(self, name: str, cls: Callable = None, force: bool = False):
        return self.register(category=LOSS_CLASS, name=name, cls=cls, force=force)

    def build_loss(self, config: dict, default_args=None):
        return self.build_from_config(
            category=LOSS_CLASS, config=config, default_args=default_args
        )

    def register_data_augmentation(
        self, name: str, cls: Callable = None, force: bool = False
    ):
        return self.register(
            category=DATA_AUGMENTATION_CLASS, name=name, cls=cls, force=force
        )

    def build_data_augmentation(self, config: dict, default_args=None):
        return self.build_from_config(
            category=DATA_AUGMENTATION_CLASS, config=config, default_args=default_args
        )


REGISTRY = Registry()
