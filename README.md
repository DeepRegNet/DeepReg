<img src="./deepreg_logo_purple.svg" alt="deepreg_logo" title="DeepReg" width="150" />
# DeepReg: deep learning for image registration

DeepReg is an open-source toolkit for research in medical image registration using deep learning. The current version is based on TensorFlow 2. This toolkit contains implementations for unsupervised- and weaky-supervised algorithms with their combinations and variants, with a practical focus on diverse clinical applications, as in provided examples.

This is still under development. However some of the functionalities can be accessed as follows.

## Setup

The easiest way to install the python environment is with Miniconda/Anaconda.
- Use `conda create --name deepreg python=3.7 tensorflow-gpu=2.2` to setup the latest gpu-support with tf.
- Use `pip install -r requirements.txt` to install the rest requirements
- Use `pip install -e .` to install the package

## Demo

### Train

The training can be launched using `deepreg_train` and it accepts the following parameters
- `-g` or `--gpu`, **required**, providing available GPU indices, e.g. `-g 0` uses GPU of index 0 and `-g "0,1"` uses GPU of index 0 and 1.
- `-c` or `--config_path`, **required**, providing the path of the configuration file, `-c demo.yaml`. Some default configuration files are provided under `src/config/`.
- `--gpu_allow_growth` providing this flag will prevent tensorflow to reserve all available GPU memory.
- `--ckpt_path` providing the checkpoint to load, to prevent start training from random initialization. The path must ended in `.ckpt`, e.g. `--ckpt logs/demo/save/weights-epoch100.ckpt`
- `--log` providing the name of log folder. It not provided, a timestamp based folder name will be used.

For example, `deepreg_train -g "" -c deepreg/config/mr_us_ddf.yaml --log demo` will launch a training using the configuration file `deepreg/config/mr_us_ddf.yaml` without GPU and the log will be saved under `logs/demo`.

### Predict

The training can be launched using `deepreg_predict` and it accepts the following parameters
- `-g` or `--gpu`, **required**, providing available GPU indices, e.g. `-g 0` uses GPU of index 0. Multi-GPU setting is not tested for prediction.
- `--mode`, **required**, providing which part of the data should be evaluated. Must be one of `train`/`valid`/`test`.
- `--ckpt_path`, **required**, providing the checkpoint to load, to prevent start training from random initialization. The path must ended in `.ckpt`, e.g. `--ckpt logs/demo/save/weights-epoch100.ckpt`. When loading the checkpoint, a backup configuration file wil be used, in the example above, the configuration file will be under `logs/demo/`. 
- `--gpu_allow_growth`, providing this flag will prevent tensorflow to reserve all available GPU memory.
- `-b` or `--batch_size`, providing the batch size, the number of data samples must be divided evenly by batch size during prediction. If not provided, batch size of 1 will be used.
- `--log` providing the name of log folder. It not provided, a timestamp based folder name will be used.

For example, `deepreg_predict -g "" --ckpt_path logs/demo/save/weights-epoch2.ckpt --mode test --log demo_test_pred` will launch a prediction on the test data using the provided checkpoint. The results will be saved under `logs` in a new folder, in which a `metric.log` will be generated. An example of `metric.log` will be like

```
image0, label 0, dice 0.0000, dist 21.1440
image0, label 1, dice 0.0000, dist 25.7530
image0, label 2, dice 0.0000, dist 16.5396
image1, label 0, dice 0.4170, dist 12.2923
image1, label 1, dice 0.0000, dist 14.9628
image1, label 2, dice 0.0000, dist 12.5561
```

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
