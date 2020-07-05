# Configuration

## Command line arguments

Once DeepReg is installed, multiple command line tools are available, including `train`
and `predict`.

### Train

The `train` accepts the following arguments:

**Required:**

- `--gpu` or `-g`, which provides the index or indices of GPUs for training.<br>
  Example: `--gpu ""` for CPU only, `--gpu "0"` for using only GPU 0, `--gpu "0,1"` for
  using GPU 0 and 1.
- `--config_path` or `-c`, which provides the configuration of the training. The path
  must end with `.yaml`.<br> Example:
  `--config_path deepreg/config/test/unpaired_labeled_ddf.yaml`<br> Optionally, multiple
  paths can be passed, and the configuration will be merged. In case of the conflicts,
  values are overwritten by the last config defining them.<br> Example:
  `--config_path deepreg/config/test/unpaired_labeled_ddf.yaml deepreg/config/test/unlabeled.yaml`
  change the data to be unlabeled.

**Optionally:**

- `--gpu_allow_growth` or `-gr`, if given then Tensorflow will not reverse all the GPU
  memory for training.<br> Default is to reserve all the GPU memory.
- `--ckpt_path` or `-k`, which provides the path of the saved model's checkpoint so that
  the training will start from the given checkpoint. The path must end with `.ckpt`.<br>
  Example: `--ckpt_path logs/test/save/weights-epoch2.ckpt`<br> Default is to start
  training from random initialization.
- `--log_dir` or `-l`, which gives the name of the directory to save logs. The directory
  will be under `logs/`.<br> Example: `--log_dir test` will save the logs under
  `logs/test/`.<br> Default is to create a timestamp based directory.

### Predict

The `predict` accepts the following arguments:

**Required:**

- `--gpu` or `-g`, which is the same as `train`.
- `--ckpt_path` or `-k`, which provides the path of the saved model's checkpoint for
  prediction. The usage is the same as `train`.
- `--config_path` or `-c`, which provides the configuration of the training. The path
  must end with `.yaml`.<br> Example:
  `--config_path deepreg/config/test/unpaired_labeled_ddf.yaml`<br> Optionally, multiple
  paths can be passed, and the configuration will be merged. In case of the conflicts,
  values are overwritten by the last config defining them.<br> Example:
  `--config_path deepreg/config/test/unpaired_labeled_ddf.yaml deepreg/config/test/unlabeled.yaml`
  change the data to be unlabeled.
- `--mode` or `-m`, which specifies on which split of the data the prediction is
  performed. It must be one of `train` / `valid` / `test`.<br> Example: `--mode test`
  will evaluate the model on test data.

**Optionally:**

- `--gpu_allow_growth` or `-gr`, which is the same as `train`.
- `--log_dir` or `-l`, which is the same as `train`.
- `--batch_size` or `-b`, which provides the batch size for prediction. The default
  value is 1.

## Configuration file

Besides the arguments provided to the command line tools, more configuration of the
training or prediction is provided in a `yaml` file. The configuration file contains two
sections, `dataset` and `train`.

The `dataset` section is for defining the dataset and corresponding loader. Read the
[dataset loader configuration](doc_data_loader.md) for more details.

The `train` section is for defining the neural network's structure, the training loss,
and other hyper-parameters for training like batch size, optimizer, and learning rate.
Read the
[example configuration](https://github.com/ucl-candi/DeepReg/blob/master/deepreg/config/unpaired_labeled_ddf.yaml)
for more details.
