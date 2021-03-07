Installation
============

DeepReg can be installed in Python 3.7 and external python dependencies are mainly defined in `requirements`_.
DeepReg primarily supports and is regularly tested with Ubuntu and Mac OS.

There are multiple different methods to install DeepReg:

1. Clone `DeepReg`_ and create a virtual environment using `Anaconda`_ / `Miniconda`_ (**recommended**).
2. Clone `DeepReg`_ and build a docker image using the provided docker file.
3. Install directly from PyPI release without cloning `DeepReg`_.

Install via Conda
-----------------

The recommended method is to install DeepReg in a dedicated virtual
environment using `Anaconda`_ / `Miniconda`_.

Please clone `DeepReg`_ first and change current directory to the DeepReg root directory:

.. code:: bash

    git clone https://github.com/DeepRegNet/DeepReg.git
    cd DeepReg

Then, install or update the conda environment following the instructions below.
Please see the `official conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
for more details.

.. tabs::

    .. tab:: Linux

        Install prerequisites (Optional).

        .. code:: bash

            sudo apt-get update
            sudo apt-get install graphviz

        Install DeepReg without GPU support.

        .. code:: bash

            conda env create -f environment_cpu.yml
            conda activate deepreg

        Install DeepReg with GPU support.

        .. code:: bash

            conda env create -f environment.yml
            conda activate deepreg

    .. tab:: Mac OS

        Install prerequisites (Optional).

        .. code:: bash

            brew install graphviz

        Install DeepReg without GPU support.

        .. code:: bash

            conda env create -f environment_cpu.yml
            conda activate deepreg

        Install DeepReg with GPU support.

        .. warning::

            Not supported or tested.

    .. tab:: Windows

        Install DeepReg without GPU support.

        .. warning::

            DeepReg on Windows is not fully supported.
            However, you can use the `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__.
            Set up WSL and follow the DeepReg setup instructions for Linux.

        Install DeepReg with GPU support.

        .. warning::

            Not supported or tested.


After activating the conda environment, please install DeepReg locally:

.. code:: bash

    pip install -e .

Install via docker
------------------

We also provide the docker file for building the docker image.
Please clone `DeepReg`_ repository first:

.. code:: bash

    git clone https://github.com/DeepRegNet/DeepReg.git

Then, install DeepReg following the instructions below.

Install docker
^^^^^^^^^^^^^^

Docker can be installed following the `official documentation <https://docs.docker.com/get-docker/>`__.

For Linux based OS, there are some `additional setup <https://docs.docker.com/engine/install/linux-postinstall/>`__ after the installation.
Otherwise you might have permission errors.

Build docker image
^^^^^^^^^^^^^^^^^^

.. code:: bash

    docker build . -t deepreg -f Dockerfile

where

- :code:`-t` names the built image as :code:`deepreg`.
- :code:`-f` provides the docker file for configuration.

Create a container
^^^^^^^^^^^^^^^^^^

.. code:: bash

    docker run --name <container_name> --privileged=true -ti deepreg bash

where
- :code:`--name` names the created container.
- :code:`--privileged=true` is required to solve the permission issue linked to TensorFlow profiler.
- :code:`-it` allows interaction with container and enters the container directly,
check more info on `stackoverflow <https://stackoverflow.com/questions/48368411/what-is-docker-run-it-flag>`__.

Remove a container
^^^^^^^^^^^^^^^^^^

.. code:: bash

    docker rm -v <container_name>

which removes a created container and its volumes, check more info on `docker documentation <https://docs.docker.com/engine/reference/commandline/rm/)>`__.

Install via PyPI
----------------

Please use the following command to install DeepReg directly from the PyPI release:

.. code:: bash

    pip install deepreg

The PyPI release currently does not ship with test data and demos.
Running examples, such as those in `Quick Start`_ and `DeepReg Demo`_,
in this documentation may require downloading additional test data.

Once you have installed DeepReg via :code:`pip`, you can run the following
command to download the necessary files to run all examples by:

.. code:: bash

    deepreg_download

The above will download the files to the current working directory.
If you need to download to a specific directory, use the
:code:`--output_dir` or :code:`-d` flag to specify this.

**Note**

1. All dependencies, APIs and command-line tools will be installed automatically via each installation method.
2. Only released versions of DeepReg are available via PyPI release.
   Therefore it is different from the `latest (unstable) version <https://github.com/DeepRegNet/DeepReg>`__ on GitHub.

.. _Quick Start: quick_start.html
.. _DeepReg Demo: ../demo/introduction.html
.. _Anaconda: https://docs.anaconda.com/anaconda/install
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _DeepReg: https://github.com/DeepRegNet/DeepReg
.. _requirements: https://github.com/DeepRegNet/DeepReg/blob/main/requirements.txt
