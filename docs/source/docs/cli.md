# Command Line Tools

With DeepReg installed, multiple command line tools are available, currently including:

- `deepreg_train`, for training a registration network.
- `deepreg_predict`, for evaluating a trained network.
- `deepreg_warp`, for warping an image with a dense displacement field.

## Train

`deepreg_train` accepts the following arguments via command line tools. More
configuration can be specified in the configuration file. Please see
[configuration file](configuration.html) for further details.

### Required arguments

- **GPU**:

  `--gpu` or `-g`, specifies the index or indices of GPUs for training.

  Example usage:

  - `--gpu ""` for CPU only
  - `--gpu "0"` for using only GPU 0
  - `--gpu "0,1"` for using GPU 0 and 1.

- **Configuration**:

  `--config_path` or `-c`, specifies the configuration file for training.

  The path must end with `.yaml`.

  Optionally, multiple paths can be specified, and the configuration will be merged. In
  case of conflicts, values are overwritten by the last config file defining them.

  Example usage:

  - `--config_path config1.yaml` for using one single configuration file.
  - `--config_path config1.yaml config2.yaml` for using multiple configuration files.

### Optional arguments

- **CPU allocation**:

  `--num_workers`, if given, TensorFlow will use limited CPUs.

  By default, it uses only 1 CPUs. Setting it to non-positive values will be using all
  CPUs.

  Example usage:

  - `--num_workers 2` for using at most 2 CPUs.

- **GPU memory allocation**:

  `--gpu_allow_growth` or `-gr`, if given, TensorFlow will only grow the memory usage as
  is needed.

  By default, it allocates all available GPU memory.

  Example usage:

  - `--gpu_allow_growth`, no extra argument is needed.

- **Load checkpoint**:

  `--ckpt_path` or `-k`, specifies the path of the saved model checkpoint, so that the
  training will be resumed from the given checkpoint.

  The path must end with `.ckpt`.

  By default, it starts training from a random initialization.

  Example usage:

  - `--ckpt_path weights-epoch2.ckpt` for reloading the given checkpoint.

- **Log directory**:

  `--log_dir`, specifies the log directory for logging output information and results.

  By default, it is `logs` under the package root.

  Example usage:

  - `--log_dir logs` for specifying the log directory `logs/` under current directory.

- **Experiment name**:

  `--exp_name` or `-n`, specifies the name of an experiment (every time a training or a
  prediction is run), which will be used together with the log directory (via `log_dir`)
  to specify the sub-folder that saves the output information and results from
  individual experiments (runs).

  If this is not provided, it creates a timestamp-named sub-folder under the `log_dir`,
  by default, e.g. `logs/20200810-194042/`.

  Example usage:

  - `--exp_name test --log_dir logs` for saving under `logs/test/`.
  - `--log_dir logs` for saving under `logs/20210101-120000/`, assuming
    `20210101-120000` is current time.
  - `--exp_name test` for saving under `DeepReg/logs/test/`, assuming `DeepReg` is the
    package root.

- **Maximum number of epochs**:

  `--max_epochs`, specifies the maximum number of epochs for training and overwrites the
  value defined in the configuration.

  By default, the value is -1, meaning the number of epochs will be defined by
  configuration.

  Example usage:

  - `--max_epochs 2` for run training only for two epochs.

### Output

During the training, multiple output files will be saved in the log directory
`logs/log_dir`, where `log_dir` is specified in the arguments, otherwise a timestamped
folder name will be used. The output files are:

- `config.yaml` is a backup of the used configuration. It can be used for prediction. In
  case of multiple configuration files, a merged configuration file will be saved.
- `train/` and `validation/` are the directories that save tensorboard logs on metrics.
- `save/` is the directory containing saved checkpoints of the trained network.

## Predict

`deepreg_predict` accepts the following arguments via command line tools. More
configuration can be specified in the configuration file. Please see
[configuration file](configuration.html) for further details.

### Required arguments

- **GPU**:

  `--gpu` or `-g`, specifies the index or indices of GPUs for training.

  Example usage:

  - `--gpu ""` for CPU only
  - `--gpu "0"` for using only GPU 0
  - `--gpu "0,1"` for using GPU 0 and 1.

- **Model checkpoint**:

  `--ckpt_path` or `-k`, specifies the path of the saved model checkpoint, so that the
  trained model will be loaded for evaluation.

  The path must end with `.ckpt`.

  Example usage:

  - `--ckpt_path weights-epoch2.ckpt` for reloading the given checkpoint.

- **Evaluation data**:

  `--split`, specifies in which data set the prediction is performed.

  It must be one of `train` / `valid` / `test`.

  Example usage:

  - `--split test` for evaluating the model on test split.

### Optional arguments

- **CPU allocation**:

  `--num_workers`, if given, TensorFlow will use limited CPUs.

  By default, it uses all available CPUs.

  Example usage:

  - `--num_workers 2` for using at most 2 CPUs.

- **GPU memory allocation**:

  `--gpu_allow_growth` or `-gr`, if given, TensorFlow will only grow the memory usage as
  is needed.

  By default, it allocates all availables in the GPU memory.

  Example usage:

  - `--gpu_allow_growth`, no extra argument is needed.

- **Log directory**:

  `--log_dir`, specifies the log directory for logging output information and results.

  By default, it is `logs` under the package root.

  Example usage:

  - `--log_dir logs` for specifying the log directory `logs/` under current directory.

- **Experiment name**:

  `--exp_name` or `-n`, specifies the name of an experiment (every time a training or a
  prediction is run), which will be used together with the log directory (via `log_dir`)
  to specify the sub-folder that saves the output information and results from
  individual experiments (runs).

  If this is not provided, it creates a timestamp-named sub-folder under the `log_dir`,
  by default, e.g. `logs/20200810-194042/`.

  Example usage:

  - `--exp_name test --log_dir logs` for saving under `logs/test/`.
  - `--log_dir logs` for saving under `logs/20210101-120000/`, assuming
    `20210101-120000` is current time.
  - `--exp_name test` for saving under `DeepReg/logs/test/`, assuming `DeepReg` is the
    package root.

- **Batch size**:

  `--batch_size` or `-b`, specifies the number of samples per step for prediction. If
  using multiple GPUs, i.e. `n` GPUs, each GPU will have mini batch size
  `batch_size / n`. Thus, `batch_size` should be divided by `n` evenly.

  The default value is 1.

  Example usage:

  - `--batch_size 2` for using a global mini-batch size of 2.

- **Save outputs in Nifti format**:

  The predicted 3D tensors can be saved in Nifti format for further calculation.

  By default, it saves outputs in Nifti format.

  Example usage:

  - `--save_nifti`, for saving the outputs in Nifti format.
  - `--no_nifti`, for not saving the outputs in Nifti format.

- **Save outputs in png format**:

  The predicted 3D tensors can be saved as a slice of 2D images for quick visualization.

  As values have to be normalized between 0~255 (or 0~1) for png files (Nifti files are
  not impacted), all images (`moving_image`, `fixed_image` and `pred_fixed_image`) and
  displacement/velocity fields (`ddf` and `dvf`) will be normalized before being saved.
  Labels (`moving_label`, `fixed_label` and `pred_fixed_label`) are not affected as they
  are already within 0~1.

  By default, it saves the outputs in png format.

  Example usage:

  - `--save_png`, for saving the outputs in png format.
  - `--no_png`, for not saving the outputs in png format.

- **Configuration**:

  `--config_path` or `-c`, specifies the configuration file for prediction.

  The path must end with `.yaml`.

  By default, it uses the configuration file saved in the directory of the given
  checkpoint.

  Example usage:

  - `--config_path config1.yaml` for using one single configuration file.

### Output

During the evaluation, multiple output files will be saved in the log directory
`logs/log_dir/mode` where

- `log_dir` is defined in arguments, or a timestamped folder name will be used;
- `mode` is `train` or `valid` or `test`, specified by the argument.

The saved files include:

- Metrics to evaluate the registration performance
  - `metrics.csv` saves the metrics on all samples. Each line corresponds to a data
    sample.
  - `metrics_stats_per_label.csv` saves the mean, median and std of each metrics on all
    samples with the same label index.
  - `metrics_stats_overall.csv` saves a set of commonly used statistics (such as mean
    and std) on the metrics over all samples.
- Inputs and predictions for each pair of image.

  Each pair has its own directory and the followings tensors are saved inside if
  available. Tensors can be saved in Nifti format (one single file) or in png format
  (one folder contains all image slices, ordered by depth) or both.

  - `ddf`, `dvf`, `affine`

    DDF stands for dense displacement field; DVF stands for dense (static) velocity
    field.

    The 12 parameters of affine transformation are saved in `affine.txt`.

  - `moving_image`, `fixed_image` and `pred_fixed_image`

    `pred_fixed_image` is the warped moving image if the network predicts a DDF or a DVF
    or an affine transformation.

  - `moving_label`, `fixed_label` and `pred_fixed_label` under directory `label_i` if
    the sample is labeled and `i` is the label index.

    `pred_fixed_label` is the predicted label in the fixed image space. In many cases,
    this is equivalent to the warped moving label, if the network predicts a DDF or a
    DVF or an affine transformation.

## Warp

`deepreg_warp` accepts the following arguments:

### Required arguments

- **Image file**:

  `--image` or `-i`, specifies the file path of the image/label.

  The image/label should be saved in a Nifti file with suffix `.nii` or `.nii.gz`. The
  image/label should be a 3D / 4D tensor, where the first three dimensions correspond to
  the moving image shape and the fourth can be a channel of features.

  Example usage:

  - `--image input_image.nii.gz`

- **DDF file**:

  `--ddf` or `-d`, specifies the file path of the DDF.

  The DDF should be saved in a Nifti file with suffix `.nii` or `.nii.gz`. The DDF
  should be a 4D tensor, where the first three dimensions correspond to the fixed image
  shape and the fourth dimension has 3 channels corresponding to x, y, z axes.

  Example usage:

  - `--image input_DDF.nii.gz`

### Optional arguments

- **Output directory**:

  `--out` or `-o`, specifies the file path for the output.

  The path should end with `.nii` or `.nii.gz`, otherwise the output path will be
  corrected automatically based on the given path.

  By default, it saves the output as `warped.nii.gz` in the current directory.

  Example usage:

  - `--out output_image.nii.gz`

### Output

The warped image is saved in the given output file path, otherwise the default file path
`warped.nii.gz` will be used.

## Visualise

In addition to the images in the output, DeepReg provides a set of tools with the
command `deepreg_vis`. See more details in
[its usage documentation](visualisation_tool.html).
