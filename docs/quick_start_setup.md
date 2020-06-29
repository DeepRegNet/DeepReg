# Setup

DeepReg is written in Python 3 (>=3.7), relying on several external libraries that
provide core IO functionalities as well as several processing tools. The dependencies
for this package are managed by `pip`.

The package is primarily distributed via Github with future support via PyPI.

## Create a virtual environment

The recommended method is to install DeepReg in a dedicated virtual environment to avoid
issues with other dependency. It can be easily created using
[Anaconda](https://docs.anaconda.com/anaconda/install/) /
[Miniconda](https://docs.conda.io/en/latest/miniconda.html):

<!-- tabs:start -->

#### ** Linux **

With CPU only

```bash
conda create --name deepreg python=3.7 tensorflow-gpu=2.2
source activate deepreg
```

With GPU

```bash
conda create --name deepreg python=3.7 # create the virtual environment
source activate deepreg # activate the environment
```

<!-- tabs:end -->

## Install the package

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
