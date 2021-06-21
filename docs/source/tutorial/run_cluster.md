# Running DeepReg Remotely (on the Cluster)

This tutorial gives an example of how to run DeepReg remotely (e.g. on a cluster). Our
example is specific to the UCL cluster, which has the operating system CentOS 7 (similar
to Ubuntu), with job scheduler Sun Grid Engine (SGE). More information on the specific
configuration at UCL is available [here](https://hpc.cs.ucl.ac.uk/job-submission/).

## Installing the Environment

Below is the script to install the environment for DeepReg in the cluster. If you want
to switch to the current working branch. Call `git branch origin/<branch_name>` after
downloading DeepReg.

```
git clone https://github.com/<personal_acount_id>/DeepReg.git
module load default/python/3.8.5

cd <DeepReg_Dir>

export PATH=/share/apps/anaconda3-5/bin:$PATH
conda env create -f environment.yml   #set up environment
source /share/apps/source_files/cuda/cuda-10.1.source    # set up cuda for GPU
source activate deepreg   # activate conda env

export CONDA_PIP="/home/<cs_account_id>/.conda/envs/deepreg/bin/pip"
$CONDA_PIP install -e .
```

`module` is a command to find packages and set up them in your own storge. You can get
more information by `module help`. Another way to install packages is

```
export PATH=/share/apps/<module_name>/bin:$PATH
```

You can find any available packages in cluster nodes by

```
ls /share/apps/ | grep '<package_name>*'
```

**Tip:** For now, all the packages are stored in `share/apps/`. If the path does not
exist, try `module load default/python/3.8.5`. Then, call `$PATH` to find the new
location of packages.

## Example Script

Below is the submission script for running quick start example
[here](../getting_started/quick_start.md). Change `<DeepReg_dir>` in the script to the
remote DeepReg repo location and save the below code in a `<your_name>.qsub`. Submit the
job with `qsub <your_name>.qsub` and check the status of the job with `qstat`, the saved
stdout and stderr is in `home/<cs_account_id>/logs/`.

```
#!/bin/bash
#$ -S /bin/bash   # bash for job
#$ -l gpu=true   # use gpu
#$ -l tmem=10G   # virtual mem used
#$ -l h_rt=36:0:0   # max job runtime hour:min:sec
#$ -N DeepReg_tst   # job name
#$ -wd home/<cs_account_id>/logs   # output, error log dir.
#Please call `mkdir logs` before using the script.

hostname
date

cd ../<DeepReg_dir>
export PATH=/share/apps/anaconda3-5/bin:$PATH
source activate deepreg   # activate conda env
export PATH=/share/apps/cuda-10.1/bin:/share/apps/gcc-8.3/bin:$PATH   # path for cuda, gcc
export LD_LIBRARY_PATH=/share/apps/cuda-10.1/lib64:/share/apps/gcc-8.3/lib64:$LD_LIBRARY_PATH   # path for cuda, gcc

deepreg_train \
--gpu \
--config_path config/unpaired_labeled_ddf.yaml \
--log_dir test
```

You also can directly access one of four cluster nodes reserved for development
purposes, by the command below. You can then run your code via the command line. More
information on the specific configuration at UCL is available
[here](https://hpc.cs.ucl.ac.uk/job-submission/).

```
qrsh -l tmem=14G,h_vmem=14G
```

## Contact and Version

Please contact stefano.blumberg.17@ucl.ac.uk, for information about the cluster. The
information here is likely to change as UCL updates the software on the cluster.
