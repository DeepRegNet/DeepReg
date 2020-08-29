# Configuration File

Besides the arguments provided to the command line tools, detailed training and
prediction configuration is specified in a `yaml` file. The configuration file contains
two sections, `dataset` and `train`.

## Dataset section

[dataset loader configuration](dataset_loader.html) for more details.

## Train section

The `train` section defines the neural network, training loss and other training
hyper-parameters, such as batch size, optimizer, and learning rate. Read the
[example configuration](https://github.com/DeepRegNet/DeepReg/blob/master/deepreg/config/unpaired_labeled_ddf.yaml)
for more details.
