# How to configure DeepReg options

(under development)

## Command line arguments

`DeepReg` is primarily a command line tool that provides two basic functions, `train` and `predict`.

`train` requires:
`-g` or `--gpu`
`-c` or `--config_path`
optionally:
`-gr` or `--gpu_allow_growth`
`-k` or `--ckpt_path`
`-l` or `--log`

`predict` requires:
`-g` or `--gpu`
`-k` or `--ckpt_path`
optionally:
`-c` or `--config_path`

## Config files

`train` requires two sections be configured in config file, `dataset` and `train`;

`predict` requires at minimum `dataset` be configured.

### Options for predefined loaders

The options for `dataset` are summarised in [Dataset configuration](./configuration_dataset.md)

### Options for training

The options for `train` are summarised in [Training configuration](./configuration_training.md)
