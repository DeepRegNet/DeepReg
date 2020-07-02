# Quick Start

## Setup

DeepReg is written in Python 3 (>=3.7), relying on several external libraries that
provide core IO functionalities as well as several processing tools. The dependencies
for this package are managed by `pip`.

The package is primarily distributed via Github with future support via PyPI.

### Create a virtual environment

The recommended method is to install DeepReg in a dedicated virtual environment to avoid
issues with other dependency. It can be easily created using
[Anaconda](https://docs.anaconda.com/anaconda/install/) /
[Miniconda](https://docs.conda.io/en/latest/miniconda.html):

<!-- tabs:start -->

#### ** Linux **

```bash
conda create --name deepreg python=3.7 tensorflow=2.2  # create the virtual environment
conda activate deepreg # activate the environment
```

<!-- tabs:end -->

### Install the package

The recommended method is to clone the repository and install it locally. All necessary
dependencies will be installed automatically.

```bash
git clone https://github.com/ucl-candi/DeepReg.git # clone the repository
pip install -e . # install the package
```

Optionally, you can install the
[master branch](https://github.com/ucl-candi/DeepReg.git) of the package directly from
the repository:

```bash
pip install git+https://github.com/ucl-candi/DeepReg.git
```

## Training

Train a registration network using unpaired and labeled test data with a predefined
configuration:

```bash
train --gpu "" --config_path deepreg/config/unpaired_labeled_ddf.yaml --log_dir test
```

where

- `--gpu ""` means not using GPU. Use `--gpu "0"` to use the GPU of index 0 and use
  `--gpu "0,1"` to use two GPUs.
- `--config_path deepreg/config/unpaired_labeled_ddf.yaml` provides the configuration
  for the training. Read configuration for more details.
- `--log_dir test` specifies the output folder, the output will be saved in `logs/test`.

## Inference

The trained network can be evaluated using unseen test data set:

```bash
predict -g "" --ckpt_path logs/test/save/weights-epoch2.ckpt --mode test
```

where

- `--gpu ""` means not using GPU.
- `--ckpt_path logs/test/save/weights-epoch2.ckpt` provides the checkpoint path of the
  trained network. A copy of training configuration is saved under `logs/test/`, so no
  configuration is required as input.
- `--mode test` means the inference is performed on the test data set. Other options can
  be `train` or `valid`.

This is a demo using data set to train a registration network. Read tutorials and
documentation for more details.
