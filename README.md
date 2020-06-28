<img src="./deepreg_logo_purple.svg" alt="deepreg_logo" title="DeepReg" width="150" />

# DeepReg: deep learning for image registration

DeepReg is an open-source toolkit for research in medical image registration using deep learning. The current version is based on TensorFlow 2. This toolkit contains implementations for unsupervised- and weaky-supervised algorithms with their combinations and variants, with a practical focus on diverse clinical applications, as in provided examples.

This is still under development. However some of the functionalities can be accessed.

## Quick start

- Create a new virtual environment using Anaconda/Miniconda:
  `conda create --name deepreg python=3.7 tensorflow-gpu=2.2`

- Install DeepReg:
  `pip install -e .`

- Train a registration network:
  `train -g "" --config_path deepreg/config/unpaired_unlabeled_ddf.yaml --log_dir test`

- Prediction using the trained registration network:
  `predict -g "" --ckpt_path logs/test/save/weights-epoch2.ckpt --mode test`

## Tutorials

### Two ways to get started with DeepReg

[Get started with image registration using deep learning](docs/tutorials/registration.md)

[Get started with demos](docs/tutorials/demo.md)

### How-to guides

[How to configure DeepReg options](docs/tutorials/configurations.md)

[How to arrange data files and folders to use predefined data loaders](docs/tutorials/predefined_loader.md)

(under development)

Other tutorial topics can be found in the wiki [Tutorial Index](https://github.com/ucl-candi/DeepReg/wiki/Tutorial-Index)

### System setup

(under development)

## Demos

(under development)

## Contributions

We welcome contributions! Please refer to the [contribution guidelines](./docs/CONTRIBUTING.md) for the toolkit.
