# How to configure DeepReg options

## Command line arguments

`DeepReg` is primarily a command line tool that provides two basic functions, `train` and `predict`.

System options are reuired to parse to these functions:
`-g` or `--gpu`
`-c` or `--config_path`
`-g` or `--gpu <str>`
`--gpu_allow_growth`
`-k` or `--ckpt_path`
`-l` or `--log`

## Config files

`train` requires two sections be configured in config file, `dataset` and `train`;
`predict` requires at minimum `dataset` be configured.

### Options for predefined loaders

The options for `dataset` are summarised in [Dataset configurations](./configurations_dataset.md)

### Options for training

The options for `train` are summarised in [Training configurations](./configurations_train.md)
