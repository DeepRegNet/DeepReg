# Installation guide

DeepReg is written in Python 3 (>=3.7), relying on several external libraries that provide core IO functionalities as well as several processing tools. The dependencies for this package are managed by `pip`.

The package is primarily distributed via Github with future support via PyPI.

## Setting up a virtual environmant
- Create a new virtual environment using Anaconda/Miniconda by calling the command `conda create --name deepreg python=3.7 tensorflow-gpu=2.2` to setup the latest gpu-support with Tensorflow. We recommend a separate virtual environment to avoid issues with other dependency trees you may have. To activate the environment, call `source activate deepreg` from the command line.

## Installing
You can install the package directly from the repository by calling:
 `pip install git+https://github.com/ucl-candi/DeepReg.git`.
This will install the [master branch](https://github.com/ucl-candi/DeepReg.git) on Github.

## Development
You can clone the repository in your current working directory and edit DeepReg code locally by calling the following command:
 `git clone https://github.com/ucl-candi/DeepReg.git`.
Then, calling:
  `pip install -e .`
within the cloned Deepreg repo will automatically install the package in your virtual environment as well as all necessary dependencies for development and usage.