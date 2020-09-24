# Configuration File

Besides the arguments provided to the command line tools, detailed training and
prediction configuration is specified in a `yaml` file. The configuration file contains
two sections, `dataset` and `train`.

## Dataset section

See the [dataset loader configuration](dataset_loader.html) for more details.

## Train section

The `train` section defines the neural network training hyper-parameters, by specifying
subsections, `model`, `loss`, `optimizer`, `preprocess` and other training
hyper-parameter, including `epochs` and `save_period`. See an
[example configuration](https://github.com/DeepRegNet/DeepReg/blob/main/config/unpaired_labeled_ddf.yaml),
with comments on the available options in each subsection.

This section is highly application-specific. More examples can be found in
[DeepReg Demos](../demo/introduction.html).
