<img src="./deepreg_logo_purple.svg" alt="deepreg_logo" title="DeepReg" width="150" />

# DeepReg: deep learning for image registration

DeepReg is an open-source toolkit for research in medical image registration using deep learning. The current version is based on TensorFlow 2. This toolkit contains implementations for unsupervised- and weaky-supervised algorithms with their combinations and variants, with a practical focus on diverse clinical applications, as in provided examples.

This is still under development. However some of the functionalities can be accessed.

## Quick start

- Create a new virtual environment using Anaconda/Miniconda:
  `conda create --name deepreg python=3.7 tensorflow-gpu=2.2`

- Install DeepReg:
  `pip install -e .`

- Train a registration network using test data:
  `train --gpu <str> --config_path <str> --gpu_allow_growth --ckpt_path <str> --log <str>`

- Prediction using a trained registration network:
  `predict --gpu <str> --mode <str> --ckpt_path <str> --gpu_allow_growth --log <str> --batch_size <int> --sample_label <str>`

## Tutorials

### Two ways to get started with DeepReg

[Get started with image registration using deep learning](./tutorials/registration.md)

[Get started with demos](./tutorials/demos.md)

### How-to guides

[How to configure DeepReg options](./tutorials/configuration.md)

[How to arrange data files and folders to use predefined data loaders](./tutorials/predefined_loader.md)

...... (under development)

Other tutorial topics can be found in the wiki [Tutorial Index](https://github.com/ucl-candi/DeepReg/wiki/Tutorial-Index)

### System setup

(under development)

## Demos

(under development)

## Contributions

We welcome contributions! Please refer to the [contribution guidelines](./docs/CONTRIBUTING.md) for the toolkit.
