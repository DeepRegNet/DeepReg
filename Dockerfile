FROM tensorflow/tensorflow:2.3.1-gpu

# install miniconda
ENV CONDA_DIR=/root/miniconda3
ENV PATH=${CONDA_DIR}/bin:${PATH}
ARG PATH=${CONDA_DIR}/bin:${PATH}
RUN apt-get update
RUN apt-get install -y wget git && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# directory for following operations
WORKDIR /app

# clone DeepReg
RUN git clone https://github.com/DeepRegNet/DeepReg.git
WORKDIR DeepReg
RUN git pull

# install conda env
RUN conda env create -f environment.yml \
    && conda init bash \
    && echo "conda activate deepreg" >> /root/.bashrc

# install deepreg
ENV CONDA_PIP="${CONDA_DIR}/envs/deepreg/bin/pip"
RUN ${CONDA_PIP} install -e .
