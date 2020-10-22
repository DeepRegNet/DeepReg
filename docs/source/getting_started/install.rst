Installation
============

DeepReg uses in Python 3.7 and external python dependencies are defined in `requirements <https://github.com/DeepRegNet/DeepReg/blob/main/requirements.txt>`__.
DeepReg primarily supports and is regularly tested with Ubuntu and Debian Linux distributions.

There are multiple different methods to install DeepReg:

1. Clone `DeepReg`_ and create a virtual environment using `Anaconda`_ / `Miniconda`_ (**recommended**).
2. Clone `DeepReg`_ and build a docker image using the provided docker file.
3. Install directly from PyPI release without cloning `DeepReg`_.

Create a virtual environment
----------------------------

The recommended method is to install DeepReg in a dedicated virtual
environment using `Anaconda`_ / `Miniconda`_ to avoid issues with other dependencies.

Please clone `DeepReg`_ first and install DeepReg following the instructions below.
For more documentation regarding the usage of conda environment,
please check the `official documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__.

.. tabs::

    .. tab:: Linux

        Install DeepReg without GPU support.

        .. code:: bash

            conda env create -f environment_cpu.yml
            conda activate deepreg

        Install DeepReg with GPU support.

        .. code:: bash

            conda env create -f environment.yml
            conda activate deepreg

        Update DeepReg without GPU support.

        .. code:: bash

            conda env update -f environment_cpu.yml


        Update DeepReg with GPU support.

        .. code:: bash

            conda env update -f environment.yml

    .. tab:: Mac OS

        Install DeepReg without GPU support.

        .. code:: bash

            conda env create -f environment_cpu.yml
            conda activate deepreg

        Update DeepReg without GPU support.

        .. code:: bash

            conda env update -f environment_cpu.yml

        Install/update DeepReg with GPU support.

        .. warning::

            Not supported or tested.

    .. tab:: Windows

        Install/update DeepReg without GPU support.

        .. warning::

            DeepReg on Windows is not fully supported.
            However, you can use the `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__.
            Set up WSL and follow the DeepReg setup instructions for Linux.

        Install/update DeepReg with GPU support.

        .. warning::

            Not supported or tested.

Use docker
----------

We also provide the docker file for building the docker image.
Please clone `DeepReg repository`_ first and install DeepReg following the instructions below.

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

Install the package directly
----------------------------

Please use the following command to install DeepReg directly from the PyPI release.

.. code:: bash

    pip install deepreg


**Note**

1. All dependencies, APIs and command-line tools will be installed automatically via either installation method.
   However, the PyPI release currently does not ship with test data and demos.
   Running examples in this documentation may require downloading test data
   and changing default paths to user-installed packages with the PyPI release.
   These examples include those in the `Quick Start`_ and `DeepReg Demo`_.
2. Only released versions of DeepReg are available via PyPI release.
   Therefore it is different from the `latest (unreleased) version <https://github.com/DeepRegNet/DeepReg>`__ on GitHub.

.. _Quick Start: quick_start.html
.. _DeepReg Demo: ../demo/introduction.html
.. _Anaconda: https://docs.anaconda.com/anaconda/install
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _DeepReg: https://github.com/DeepRegNet/DeepReg
