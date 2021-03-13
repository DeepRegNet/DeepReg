# Registry

DeepReg adopts the
[registry system](https://github.com/DeepRegNet/DeepReg/blob/main/deepreg/registry.py)
to facilitate the addition of custom functionalities.

## Description

The class `Registry` maintains a dictionary mapping `(category, key)` to `value`, where

- `category` is the class category, e.g. `"backbone_class""` for backbone classes.
- `key` is the name of the registered class, e.g. `"unet"` for the class `UNet`.
- `value` is the registered class, e.g. `UNet` corresponding to `"unet"`.

A global variable `REGISTRY = Registry()` is defined to provide a central control of all
classes. The supported categories and the registered classes are resumed in the
[registered classes](registered_classes.html) page.

### Register a class

To register a class into `REGISTRY`, it is recommended to use `register` as a
**decorator**. Consider the `UNet` class for backbone as an example.

```python
from deepreg.registry import REGISTRY


@REGISTRY.register(category="backbone_class", name="unet")
class UNet:
    """UNet-style backbone."""
```

The decorator automatically registers the class upon import. To ensure that this class
is registered when `import deepreg` is called, this class and related parent modules
need to be imported in the `__init__.py` files.

For the purpose of code simplicity, a specific register function is defined for each
category. For instance, we can use `register_backbone` for `UNet`:

```python
from deepreg.registry import REGISTRY


@REGISTRY.register_backbone(name="unet")
class UNet:
    """UNet-style backbone."""
```

### Instantiate a class

To instantiate a registered example in `REGISTRY`, we call `build_from_config` , which
allows creating a class instance from a config directly. The config should be a
dictionary containing the key `name` and other required keys for the class. Please check
the [configuration documentation](configuration.html) and the docstring of the related
classes for detailed configuration requirements. The path of the registered classes are
resumed in the [registered classes](registered_classes.html) page.

For instance, to instantiate a UNet class,

```python
from deepreg.registry import REGISTRY

config = dict(name="unet",
              image_size=(16,16,16),
              out_channels=3,
              num_channel_initial=2,
              out_kernel_initializer="he_normal",
              out_activation="")
unet = REGISTRY.build_from_config(category="backbone_class", config=config)
```

where `kwargs` represents other required arguments.

Similarly, for the purpose of code simplicity, a specific build function is defined for
each category. For instance, we can use `build_backbone` for `UNet`:

```python
from deepreg.registry import REGISTRY


config = dict(name="unet",
              image_size=(16,16,16),
              out_channels=3,
              num_channel_initial=2,
              out_kernel_initializer="he_normal",
              out_activation="")
unet = REGISTRY.build_backbone(config=config)
```

## Example usages

Apart from the table of [registered classes](registered_classes.html), to further
explain how to use `REGISTRY` for using customized classes, detailed examples are
provided for the following categories:

- backbone
- loss

### Custom backbone

To register a custom backbone class, the steps are as follows

1. Subclass the `Backbone` and implement a custom backbone class.
2. Import `REGISTRY` and use the decorator `@REGISTRY.register_backbone` to register the
   custom class.
3. Use the registered name in the config for using the registered custom backbone.

Please check the self-contained
[example script](https://github.com/DeepRegNet/DeepReg/blob/main/examples/custom_backbone.py)
for further details.

### Custom Loss

To register a custom loss class for images and labels, the steps are as follows

1. Subclass the `tf.keras.losses.Loss` and implement a custom backbone class.
2. Import `REGISTRY` and use the decorator `@REGISTRY.register_loss` to register the
   custom class.
3. Use the registered name in the config for using the registered custom loss.

Please check the self-contained
[example script](https://github.com/DeepRegNet/DeepReg/blob/main/examples/custom_image_label_loss.py)
for further details. There is also a more complicated
[example of parameterized custom loss](https://github.com/DeepRegNet/DeepReg/blob/main/examples/custom_parameterized_image_label_loss.py).
