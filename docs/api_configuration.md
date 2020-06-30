# How to configure DeepReg options

(under development)

## Command line arguments

`DeepReg` is primarily a command line tool that provides two basic functions, `train`
and `predict`.

`train` requires

- `-g` or `--gpu`
- `-c` or

optionally

- `--config_path`
- `--gpu_allow_growth`
- `-k` or `--ckpt_path`
- `-l` or `--log`

## Config files

`train` requires two sections be configured in config file, `dataset` and `train`;

`predict` requires at minimum `dataset` be configured.

### Options for predefined loaders

The options for `dataset` are summarised in
[Dataset configuration](tutorial/configurations_dataset.md)

### Options for training

The options for `train` are summarised in
[Training configuration](tutorial/configurations_train.md)
