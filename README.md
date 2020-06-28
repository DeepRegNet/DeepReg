<img src="./deepreg_logo_purple.svg" alt="deepreg_logo" title="DeepReg" width="150" />

# DeepReg

[![Build Status](https://travis-ci.org/ucl-candi/DeepReg.svg?branch=master)](https://travis-ci.org/ucl-candi/DeepReg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DeepReg is an open-source toolkit for research in medical image registration using deep
learning. The current version bases on TensorFlow 2. This toolkit contains
implementations for unsupervised- and weaky-supervised algorithms with their
combinations and variants, with a practical focus on diverse clinical applications, as
in provided examples.

This is still under development. However, some functionalities can be accessed already.

## Quick start

- Create a new virtual environment using
  [Anaconda](https://docs.anaconda.com/anaconda/install/) /
  [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

  `conda create --name deepreg python=3.7 tensorflow-gpu=2.2`

- Install the DeepReg package:

  `pip install -e .`

- Train a registration network using unpaired and labeled data and a predefined
  configuration:

  `train -g "" --config_path deepreg/config/unpaired_labeled_ddf.yaml --log_dir test`

- Make prediction using the trained registration network on the test data set:

  `predict -g "" --ckpt_path logs/test/save/weights-epoch2.ckpt --mode test`

## Documentation

Read the documentation at [DeepReg](https://ucl-candi.github.io/DeepReg/).
