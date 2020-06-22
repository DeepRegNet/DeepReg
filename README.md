<img src="./deepreg_logo_purple.svg" alt="deepreg_logo" title="DeepReg" width="150" />  

# DeepReg: deep learning for image registration

DeepReg is an open-source toolkit for research in medical image registration using deep learning. The current version is based on TensorFlow 2. This toolkit contains implementations for unsupervised- and weaky-supervised algorithms with their combinations and variants, with a practical focus on diverse clinical applications, as in provided examples.

This is still under development. However some of the functionalities can be accessed as follows.

## Contributions
We welcome contributions! Please refer to the [contribution guidelines](https://github.com/ucl-candi/DeepReg/blob/master/docs/CONTRIBUTING.md) for the toolkit.

## Setup

The dependencies for this package are managed by `pip`.
- Create a new virtual environment using Anaconda/Miniconda by calling the command `conda create --name deepreg python=3.7 tensorflow-gpu=2.2` to setup the latest gpu-support with Tensorflow.
- Call `pip install -e .` within the Deepreg folder to install the package in the virtual environment. All necessary requirements for development and/or usage will be automatically installed.

## Command line interface applications

Deepreg supports 3 command line interface applications to facilitate training of a deep learning registration models, prediction from pre-trained models, and to generate Tensorflow records.

* **Training**: `train --gpu <str> --config_path <str> --gpu_allow_growth --ckpt_path <str> --log <str>`

* **Predict**: provided a trained, checkpointed Tensorflow model (format `.ckpt`), predictions for a given split of the data (`train`/`valid`/`test`) can be evaluated and logged. The data is taken from the config extracted from the ckpt path provided like:


`predict --gpu <str> --mode <str> --ckpt_path <str> --gpu_allow_growth --log <str> --batch_size <int> --sample_label <str>`

* **Generating Tensorflow records**: provided a config file stored in a path, the data is converted to a TFRecord type which facilitates speedier training.

`gen_tfrecord --config_path <str> --examples_per_tfrecord <int>`

Arguments: 
- `--gpu` | `-g`, string, indicating which GPU to use in training. Can pass multiple GPUs to this argument like `--gpu 0 1`
- `--config_path` | `-c` , string, path to the configuraton file in training, `.yaml` extension.
- `--gpu_allow_growth` | `-gr`, providing this flag will prevent Tensorflow from reserving all available GPU memory for given GPU(s). Defaults to False.
- `--ckpt_path` | `-k` , string, path to the checkpointed model, `.ckpt` extension.
- `--log` | `-l` , string, path where logs are to be saved. If not provided, a folder in the current working directory is generated based on a timestamp for the command.
- `--mode` | `-m`, string, which split of data to be used for prediction. One of `train`/`valid`/`test`.
- `--batch_size` | `-b`, int, number of batches to feed to the model for prediction. Defaults to 1.
- `--sample_label` | `-s`, string, currently not supported
- `--examples_per_tfrecord` | `-n`, int, number of TFRecord examples to generate at a time. Default is 64.

## Development

### Data

To use this package with a custom dataset

1. Create a folder under `deepreg/data/`, e.g. `deepreg/data/custom/`.

2. Create a new sample configuration file under `deepreg/config/`, e.g. `deepreg/config/custom.yaml`.

   There is no need to change the `tf` part, and all data related configurations should be under `data` part 
   and the only required attribute is `name`. 

3. Write a new data loader to load the custom data.

    Each data sample consists of `((moving_image`, `fixed_image`, `moving_label`, `indices`), `fixed_image)` where

    - images and labels are all assumed to be 3D single-channel images and of shape `[dim1, dim2, dim3]`.
    - indices are of shape `[num_indices]`, used for identifying the data sample. 

    The interface `DataLoader` in `deepreg/data/loader.py` defined all required functions:
    - `get_dataset` which returns a not-batched dataset and will be batched and preprocessed in `get_dataset_and_preprocess`.
    - `split_indices` which splits the indices into `image_index` and `label_index`,
        where `label_index` must be a integer and `image_index` can be an integer or a tuple. This function is used for prediction only.
    - `image_index_to_dir` which format the image index into a string. This function is used for prediction only.
     
    If generator is used to TF dataset, a more detailed interface `GeneratorDataLoader` is also provided,
    instead of defining `get_dataset`, `get_generator` is required to be defined, which needs to build a generator to return data.
   
4. Write a new load function `get_data_loader` which returns the data loader given the configuration and the mode (`train`/`valid`/`test`).

   Call `get_data_loader` inside `deepreg/data/load.py` for the custom data.

### Model

To add a custom backbone network

1. Create a file under `deepreg/modal/backbone/`, e.g. `deepreg/modal/backbone/nn.py`

2. Use the network inside `build_backbone` of `deepreg/modal/network.py`.

The network should

- receives a single input tensor of shape `[batch, dim1, dim2, dim3, in_channels]`

- outputs a single output tensor of shape `[batch, dim1, dim2, dim3, out_channels]`
