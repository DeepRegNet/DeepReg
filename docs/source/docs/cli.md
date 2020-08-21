# Command line tools

Once DeepReg is installed, multiple command line tools are available, currently
including `deepreg_train`, `deepreg_predict` and `deepreg_warp`.

## `deepreg_train`

`deepreg_train` accepts the following arguments:

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

## `deepreg_predict`

`deepreg_predict` accepts the following arguments:

**Required:**

- `--gpu` or `-g`, the same as with `deepreg_train`.
- `--ckpt_path` or `-k`, specifies the path to save model checkpoint(s) for prediction.
  The usage is the same as with `deepreg_train`.
- `--config_path` or `-c`, specifies the configuration file for training. The path must
  end with `.yaml`.<br> Example:
  `--config_path deepreg/config/test/unpaired_labeled_ddf.yaml`<br> Optionally, multiple
  paths can be specified, and the configuration will be merged. In case of conflicts,
  values are overwritten by the last config file defining them.
- `--mode` or `-m`, specifies on which data set the prediction is performed. It must be
  one of `train` / `valid` / `test`.<br> Example: `--mode test` evaluates the model on
  test data.

**Optional:**

- `--gpu_allow_growth` or `-gr`, the same as with `deepreg_train`.
- `--log_dir` or `-l`, the same as with `deepreg_train`.
- `--batch_size` or `-b`, specifies the batch size for prediction. The default is 1.
- `--save_nifti`, saves the outputs in nifti form. This is default behavior. Use
  `--no_nifti` to disable the saving.
- `--save_png`, saves the outputs in png form. By default pngs are not saved. Use
  `--save_png` to enable the saving and `--no_png` to disable the saving.
- `--config_path` or `-c`, specifies the configuration file for prediction. The path
  must end with `.yaml`. The default will be to use the saved config in the directory of
  the given checkpoint.

## `deepreg_warp`

`deepreg_warp` accepts the following arguments:

**Required:**

- `--image` or `-i`, specifies the file path of the image/label. The image/label should
  be saved in a nifti file with suffix `.nii` or `.nii.gz`. The image/label should be a
  3D / 4D tensor, where the first three dimensions correspond to the moving image shape
  and the fourth can be a channel of features.
- `--ddf` or `-d`, specifies the file path of the ddf. The ddf should be saved in a
  nifti file with suffix `.nii` or `.nii.gz`. The ddf should be a 4D tensor, where the
  first three dimensions correspond to the fixed image shape and the fourth dimension
  has 3 channels corresponding to x, y, z axes.

**Optional:**

- `--out` or `-o`, specifies the file path for the output. If this argument is not
  provided, the output will be saved as `warped.nii.gz` in the current directory. If it
  is provided, it should end with `.nii` or `.nii.gz`, otherwise the output path will be
  corrected automatically based on the given path.

_In addition to the arguments provided with the command line tools, detailed training
and prediction configuration is specified in a yaml file. Please see
[configuration file](configuration.md) for further details._
