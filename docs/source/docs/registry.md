# Registry

> This is a functionality under active development. A full documentation will be
> released later.

DeepReg is adapting the usage of the registry system recently, to facilitate the usage
of custom functionalities including

- neural network architecture
- loss
- optimizer
- data pre-processing
- data loader

The registry is defined in `deepreg/registry.py`, where the class `Registry` maintains a
dictionary mapping `(category, key)` to `value`. It also provides the
`build_from_config` functionality, which allows creating a class instance from a config
directly. This allows the simplification of the config so that each configuration file
need to provide the name and necessary arguments.

With the registry, when developing new classes **inside DeepReg**, we should use the
corresponding decorator, so that the class is registered, cf.
`deepreg/model/backbone/u_net.py` as an example. Moreover, the corresponding
`__init__.py` (maybe multiple ones) should be modified so that these classes will be
automatically registered when executing `import deepreg`. For defining custom classes
outside DeepReg, more detailed tutorial will be released later.

For now, we only support custom classes of

- backbone
- loss
