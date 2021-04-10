from copy import deepcopy
from typing import Any, Callable, Dict, Optional

BACKBONE_CLASS = "backbone_class"
LOSS_CLASS = "loss_class"
METRIC_CLASS = "metric_class"
MODEL_CLASS = "model_class"
DATA_AUGMENTATION_CLASS = "da_class"
DATA_LOADER_CLASS = "data_loader_class"
FILE_LOADER_CLASS = "file_loader_class"

KNOWN_CATEGORIES = [
    BACKBONE_CLASS,
    LOSS_CLASS,
    MODEL_CLASS,
    DATA_AUGMENTATION_CLASS,
    DATA_LOADER_CLASS,
    FILE_LOADER_CLASS,
]


class Registry:
    """
    Registry maintains a dictionary which maps `(category, key)` to `value`.

    Multiple __init__.py files have been modified so that the classes are registered
    when executing:

    .. code-block:: python

        from deepreg.registry import REGISTRY

    References:

    - https://github.com/ray-project/ray/blob/00ef1179c012719a17c147a5c3b36d6bdbe97195/python/ray/tune/registry.py#L108
    - https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/builder.py
    - https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py
    - https://towardsdatascience.com/whats-init-for-me-d70a312da583
    """

    def __init__(self):
        """Init registry with empty dict."""
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

    def contains(self, category: str, key: str) -> bool:
        """
        Verify if the key has been registered for the category.

        :param category: category name.
        :param key: value name.
        :return: `True` if registered.
        """
        return (category, key) in self._dict

    def get(self, category: str, key: str) -> Callable:
        """
        Return the registered class.

        :param category: category name.
        :param key: value name.
        :return: registered value.
        """
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
        self, category: str, config: Dict, default_args: Optional[dict] = None
    ) -> Any:
        """
        Build a class instance from config dict.

        :param category: category name.
        :param config: a dict which must contain the key "name".
        :param default_args: optionally some default arguments.
        :return: the instantiated class.
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
        """Deep copy the registry."""
        copied = Registry()
        copied._dict = deepcopy(self._dict)
        return copied

    def register_model(
        self, name: str, cls: Callable = None, force: bool = False
    ) -> Callable:
        """
        Register a model class.

        :param name: model name
        :param cls: model class
        :param force: whether overwrite if already registered
        :return: the registered class
        """
        return self.register(category=MODEL_CLASS, name=name, cls=cls, force=force)

    def build_model(self, config: Dict, default_args: Optional[dict] = None) -> Any:
        """
        Instantiate a registered model class.

        :param config: config having key `name`.
        :param default_args: optionally some default arguments.
        :return: a model instance
        """
        return self.build_from_config(
            category=MODEL_CLASS, config=config, default_args=default_args
        )

    def register_backbone(
        self, name: str, cls: Callable = None, force: bool = False
    ) -> Callable:
        """
        Register a backbone class.

        :param name: backbone name
        :param cls: backbone class
        :param force: whether overwrite if already registered
        :return: the registered class
        """
        return self.register(category=BACKBONE_CLASS, name=name, cls=cls, force=force)

    def build_backbone(self, config: Dict, default_args: Optional[dict] = None) -> Any:
        """
        Instantiate a registered backbone class.

        :param config: config having key `name`.
        :param default_args: optionally some default arguments.
        :return: a backbone instance
        """
        return self.build_from_config(
            category=BACKBONE_CLASS, config=config, default_args=default_args
        )

    def register_loss(
        self, name: str, cls: Callable = None, force: bool = False
    ) -> Callable:
        """
        Register a loss class.

        :param name: loss name
        :param cls: loss class
        :param force: whether overwrite if already registered
        :return: the registered class
        """
        return self.register(category=LOSS_CLASS, name=name, cls=cls, force=force)

    def build_loss(self, config: Dict, default_args: Optional[dict] = None) -> Callable:
        """
        Instantiate a registered loss class.

        :param config: config having key `name`.
        :param default_args: optionally some default arguments.
        :return: a loss instance
        """
        return self.build_from_config(
            category=LOSS_CLASS, config=config, default_args=default_args
        )

    def register_data_loader(
        self, name: str, cls: Callable = None, force: bool = False
    ) -> Callable:
        """
        Register a data loader class.

        :param name: loss name
        :param cls: loss class
        :param force: whether overwrite if already registered
        :return: the registered class
        """
        return self.register(
            category=DATA_LOADER_CLASS, name=name, cls=cls, force=force
        )

    def build_data_loader(
        self, config: Dict, default_args: Optional[dict] = None
    ) -> Any:
        """
        Instantiate a registered data loader class.

        :param config: config having key `name`.
        :param default_args: optionally some default arguments.
        :return: a loss instance
        """
        return self.build_from_config(
            category=DATA_LOADER_CLASS, config=config, default_args=default_args
        )

    def register_data_augmentation(
        self, name: str, cls: Callable = None, force: bool = False
    ) -> Callable:
        """
        Register a data augmentation class.

        :param name: data augmentation name
        :param cls: data augmentation class
        :param force: whether overwrite if already registered
        :return: the registered class
        """
        return self.register(
            category=DATA_AUGMENTATION_CLASS, name=name, cls=cls, force=force
        )

    def register_file_loader(
        self, name: str, cls: Callable = None, force: bool = False
    ) -> Callable:
        """
        Register a file loader class.

        :param name: loss name
        :param cls: loss class
        :param force: whether overwrite if already registered
        :return: the registered class
        """
        return self.register(
            category=FILE_LOADER_CLASS, name=name, cls=cls, force=force
        )

    def build_data_augmentation(
        self, config: Dict, default_args: Optional[dict] = None
    ) -> Callable:
        """
        Instantiate a registered data augmentation class.

        :param config: config having key `name`.
        :param default_args: optionally some default arguments.
        :return: a data augmentation instance
        """
        return self.build_from_config(
            category=DATA_AUGMENTATION_CLASS, config=config, default_args=default_args
        )


REGISTRY = Registry()
