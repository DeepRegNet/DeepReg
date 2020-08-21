Installation
============

DeepReg is written in Python 3.7. Dependent external libraries
include those provide core IO functionalities and other standard
processing tools. The dependencies for this package are managed by
``pip``.

The package is primarily distributed via Github with future support via
PyPI.

Create a virtual environment
----------------------------

The recommended method is to install DeepReg in a dedicated virtual
environment to avoid issues with other dependencies. The conda
enviroment is recommended:
`Anaconda <https://docs.anaconda.com/anaconda/install/>`__ /
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__:

DeepReg is primarily supported and regularly tested with Linux distros
Ubuntu/Debian.

.. tabs::

    .. tab:: Linux

        With CPU only

        .. code:: bash

            conda create --name deepreg python=3.7 tensorflow=2.2
            conda activate deepreg

        With GPU

        .. code:: bash

            conda create --name deepreg python=3.7 tensorflow-gpu=2.2
            conda activate deepreg


    .. tab:: Mac OS

        With CPU only

        .. code:: bash

            conda create --name deepreg python=3.7 tensorflow=2.2
            conda activate deepreg


        With GPU

        .. warning::

            Not supported or tested.

    .. tab:: Windows

        With CPU only

        .. warning::

            DeepReg is not fully supported under Windows, however,
            [Windows Subsystem Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
            may be recommended for use with CPU only. Then follow the instructions with Linux.

        With GPU

        .. warning::

            Not supported or tested.


Install the package
-------------------

The recommended method is to clone the repository and install it
locally. All necessary dependencies will be installed automatically.

.. code:: bash

    git clone https://github.com/DeepRegNet/DeepReg.git # clone the repository
    pip install -e . # install the package

Optionally, you can install the `master
branch <https://github.com/DeepRegNet/DeepReg.git>`__ of the package
directly from the repository:

.. code:: bash

    pip install git+https://github.com/DeepRegNet/DeepReg.git
