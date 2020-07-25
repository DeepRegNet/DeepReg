# Quick Start

## Setup

DeepReg is written in Python 3 (>=3.7). Dependent external libraries include those
provide core IO functionalities and other standard processing tools. The dependencies
for this package are managed by `pip`.

The package is primarily distributed via Github with future support via PyPI.

### Create a virtual environment

The recommended method is to install DeepReg in a dedicated virtual environment to avoid
issues with other dependencies. The conda enviroment is recommended:
[Anaconda](https://docs.anaconda.com/anaconda/install/) /
[Miniconda](https://docs.conda.io/en/latest/miniconda.html):

DeepReg is primarily supported and regularly tested with Linux distros Ubuntu/Debian.

<!-- tabs:start -->

#### ** Linux **

With CPU only

```bash
conda create --name deepreg python=3.7 tensorflow=2.2
conda activate deepreg # Activate the environment
```

With GPU

```bash
conda create --name deepreg python=3.7 tensorflow-gpu=2.2 # Use conda for nvidia related packages
conda activate deepreg # Activate the environment
```

#### ** MacOS **

With CPU only

```bash
conda create --name deepreg python=3.7 tensorflow=2.2
conda activate deepreg # Activate the environment
```

With GPU

:warning: Not supported or tested.

#### ** Windows **

With CPU only

:warning: DeepReg is not fully supported under Windows, however,
[Windows Subsystem Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
may be recommended for use with CPU only. Then follow the instructions with Linux.

With GPU

:warning: Not supported or tested.

<!-- tabs:end -->

### Install the package

The recommended method is to clone the repository and install it locally. All necessary
dependencies will be installed automatically.

```bash
git clone https://github.com/DeepRegNet/DeepReg.git # clone the repository
pip install -e . # install the package
```

Optionally, you can install the
[master branch](https://github.com/DeepRegNet/DeepReg.git) of the package directly from
the repository:

```bash
pip install git+https://github.com/DeepRegNet/DeepReg.git
```

## Train

Train a registration network using unpaired and labeled example data with a predefined
configuration:

```bash
train --gpu "" --config_path deepreg/config/unpaired_labeled_ddf.yaml --log_dir test
```

where

- `--gpu ""` indicates using CPU. `--gpu "0"` uses the GPU of index 0 and `--gpu "0,1"`
  uses two GPUs.
- `--config_path deepreg/config/unpaired_labeled_ddf.yaml` provides the configuration
  for the training. Read configuration for more details.
- `--log_dir test` specifies the output folder, the output will be saved in `logs/test`.

## Predict

The trained network can be evaluated using unseen example test dataset:

```bash
predict -g "" --ckpt_path logs/test/save/weights-epoch2.ckpt --mode test
```

where

- `--gpu ""` indicates using CPU for inference.
- `--ckpt_path logs/test/save/weights-epoch2.ckpt` provides the checkpoint path of the
  trained network. As a copy of training configuration is saved under `logs/test/`
  during training, so no configuration is required as input in this case.
- `--mode test` indicates the inference on the test dataset. Other options include
  `train` or `valid`.

This is a demo using example dataset to train a registration network. Read tutorials and
documentation for more details.
