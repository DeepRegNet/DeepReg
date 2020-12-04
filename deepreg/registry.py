from typing import Any

from deepreg.model.backbone.global_net import GlobalNet
from deepreg.model.backbone.local_net import LocalNet
from deepreg.model.backbone.u_net import UNet

BACKBONE_CLASS = "backbone_class"
LOSS_CLASS = "loss_class"
MODEL_CLASS = "model_class"
KNOWN_CATEGORIES = [BACKBONE_CLASS, LOSS_CLASS, MODEL_CLASS]


class Registry:
    """
    Registry maintains a dictionary which maps (category, key) to value.

    The design used the registry in ray as reference.
    https://github.com/ray-project/ray/blob/00ef1179c012719a17c147a5c3b36d6bdbe97195/python/ray/tune/registry.py#L108
    """

    def __init__(self):
        self._dict = {}
        self.register_defaults()

    def register_defaults(self):
        self.register(BACKBONE_CLASS, "local", LocalNet)
        self.register(BACKBONE_CLASS, "global", GlobalNet)
        self.register(BACKBONE_CLASS, "unet", UNet)

    def register(self, category: str, key: str, value: Any):
        """
        Registers the value with the registry.
        """
        # sanity check
        if category not in KNOWN_CATEGORIES:
            raise ValueError(
                f"Unknown category {category} not among {KNOWN_CATEGORIES}"
            )
        if self.contains(category=category, key=key):
            raise ValueError(
                f"Key {key} in category {category} has been registered with {self._dict[(category, key)]}"
            )
        # register value
        self._dict[(category, key)] = value

    def contains(self, category: str, key: str):
        return (category, key) in self._dict

    def get(self, category, key):
        if self.contains(category=category, key=key):
            return self._dict[(category, key)]
        raise ValueError(f"Key {key} in category {category} has not been registered.")

    def register_backbone(self, key, value):
        self.register(category=BACKBONE_CLASS, key=key, value=value)

    def get_backbone(self, key):
        return self.get(category=BACKBONE_CLASS, key=key)
