# Running DeepReg on the Cluster
This tutorial gives an example of how to run DeepReg remotely (e.g. on the cluster).  Our example is specific to the UCL cluster, which has operating system CentOS 7 (similar to Ubuntu), with job scheduler Sun Grid Engine (SGE).  More information on the specific configuration at UCL is available [here](https://hpc.cs.ucl.ac.uk/job-submission/).

# Running DeepReg on the Cluster
Install the environment in the cluster, as described [here](../../../README.md).  In the case you do not have root access (as the case of the UCL cluster), you might need to use pip -u option to install the requirements.txt file.

# Example Script
Below is the submission script for running quick start example [here](../getting_started/quick_start.md).  Change `<DeepReg_dir>` in the script to the DeepReg repo location and save the below code in a `<your_name>.qsub`.  Submit the job with `qsub <your_name>.qsub` and check the status of the job with `qstat`, the saved stdout and stderr is in `<DeepReg_dir>/logs/`.

```
#!/bin/bash
#$ -S /bin/bash   # bash for job
#$ -l gpu=true   # use gpu
#$ -l tmem=10G   # virtual mem used
#$ -l h_rt=36:0:0   # max job runtime
#$ -R y
#$ -N DeepReg_tst   # job name
#$ -wd <DeepReg_dir>/logs   # output, error log dir

hostname
date

cd <DeepReg_dir>
conda activate deepreg   # activate conda env
export PATH=/share/apps/cuda-10.1/bin:/share/apps/gcc-8.3/bin:$PATH   # path for cuda, gcc
export LD_LIBRARY_PATH=/share/apps/cuda-10.1/lib64:/share/apps/gcc-8.3/lib64:$LD_LIBRARY_PATH   # path for cuda, gcc

deepreg_train \
--gpu "0" \
--config_path config/unpaired_labeled_ddf.yaml \
--log_dir test
```

# Contact and Version
Please contact stefano.blumberg.17@ucl.ac.uk, for information about this questions about the cluster.  Written on 2020-10-11.
