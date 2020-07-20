# Configuration

## Command line arguments

Once DeepReg is installed, multiple command line tools are available, including `train`
and `predict`.

### Train

`train` accepts the following arguments:

**Required:**

- `--gpu` or `-g`, specifies the index or indices of GPUs for training.<br> Example:
  `--gpu ""` for CPU only, `--gpu "0"` for using only GPU 0, `--gpu "0,1"` for using GPU
  0 and 1.
- `--config_path` or `-c`, specifies the configuration file for training. The path must
  end with `.yaml`.<br> Example:
  `--config_path deepreg/config/test/unpaired_labeled_ddf.yaml`<br> Optionally, multiple
  paths can be specified, and the configuration will be merged. In case of conflicts,
  values are overwritten by the last config file defining them.<br> Example:
  `--config_path deepreg/config/test/unpaired_labeled_ddf.yaml deepreg/config/test/unlabeled.yaml`
  changes the data to be unlabeled.

**Optional:**

- `--gpu_allow_growth` or `-gr`, if given, TensorFlow will only grow the memory usage as
  is needed by the training process.<br> Default is to allocate all available GPU
  memory.
- `--ckpt_path` or `-k`, specifies the path of the saved model checkpoint, such that the
  training may be resumed from the given checkpoint. The path must end with `.ckpt`.<br>
  Example: `--ckpt_path logs/test/save/weights-epoch2.ckpt`<br> Default is to start
  training from random initialization.
- `--log_dir` or `-l`, specifies the directory name to save logs. The directory will be
  under `logs/`.<br> Example: `--log_dir test` will save the logs under `logs/test/`.
  <br> Default is to create a timestamp-named directory.

### Predict

`predict` accepts the following arguments:

**Required:**

- `--gpu` or `-g`, the same as with `train`.
- `--ckpt_path` or `-k`, specifies the path to save model checkpoint(s) for prediction.
  The usage is the same as with `train`.
- `--config_path` or `-c`, specifies the configuration file for training. The path must
  end with `.yaml`.<br> Example:
  `--config_path deepreg/config/test/unpaired_labeled_ddf.yaml`<br> Optionally, multiple
  paths can be specified, and the configuration will be merged. In case of conflicts,
  values are overwritten by the last config file defining them.
- `--mode` or `-m`, specifies on which data set the prediction is performed. It must be
  one of `train` / `valid` / `test`.<br> Example: `--mode test` evaluates the model on
  test data.

**Optional:**

- `--gpu_allow_growth` or `-gr`, the same as with `train`.
- `--log_dir` or `-l`, the same as with `train`.
- `--batch_size` or `-b`, specifies the batch size for prediction. The default is 1.
- `--config_path` or `-c`, specifies the configuration file for prediction. The path
  must end with `.yaml`. The default will be to use the saved config in the directory of
  the given checkpoint.

## Configuration file

Besides the arguments provided to the command line tools, detailed training and
prediction configuration is specified in a `yaml` file. The configuration file contains
two sections, `dataset` and `train`.

The `dataset` section defines the dataset and corresponding loader. Read the
[dataset loader configuration](doc_data_loader.md) for more details.

The `train` section defines the neural network, training loss and other training
hyper-parameters, such as batch size, optimizer, and learning rate. Read the
[example configuration](https://github.com/DeepRegNet/DeepReg/blob/master/deepreg/config/unpaired_labeled_ddf.yaml)
for more details.
