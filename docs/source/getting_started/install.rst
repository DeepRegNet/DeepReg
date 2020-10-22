Installation
============

DeepReg is written in Python 3.7. Dependent external libraries provide core IO functionalities and other standard
processing tools. The dependencies for DeepReg are managed by
``pip``.

Create a virtual environment
----------------------------

The recommended method is to install DeepReg in a dedicated virtual
environment to avoid issues with other dependencies. The conda
environment is recommended:

`Anaconda <https://docs.anaconda.com/anaconda/install/>`__ /
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__

DeepReg primarily supports and is regularly tested with Ubuntu and Debian Linux distributions.

.. tabs::

    .. tab:: Linux

        With CPU only

        .. code:: bash

            conda env create -f environment_cpu.yml
            conda activate deepreg

        With GPU

        .. code:: bash

            conda env create -f environment.yml
            conda activate deepreg


    .. tab:: Mac OS

        With CPU only

        .. code:: bash

            conda env create -f environment_cpu.yml
            conda activate deepreg


        With GPU

        .. warning::

            Not supported or tested.

    .. tab:: Windows

        With CPU only

        .. warning::

            DeepReg on Windows is not fully supported. However, you can use the `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__
            with CPU only. Set up WSL and follow the DeepReg setup instructions for Linux.

        With GPU

        .. warning::

            Not supported or tested.

Use docker
----------

We also provide the docker file for building the docker image.

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

**Install from the cloned local project**

.. code:: bash

    git clone https://github.com/DeepRegNet/DeepReg.git # clone the repository
    cd DeepReg # change working directory to the DeepReg root directory
    pip install -e . # install the package


**Install from the PyPI release**

.. code:: bash

    pip install deepreg


**Note**

All dependencies, APIs and command-line tools will be installed automatically via either installation method.
However, the PyPI release currently does not ship with test data and demos.
Running examples in this documentation may require downloading test data and changing default paths to user-installed packages with the PyPI release.
These examples include those in the `Quick Start`_ and `DeepReg Demo`_.


.. _Quick Start: quick_start.html
.. _DeepReg Demo: ../demo/introduction.html
