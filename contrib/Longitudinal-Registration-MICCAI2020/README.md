### Official inplementation for MICCAI 2020 paper: Longitudinal image registration with
temporal-order and subject-specificity discrimination

<img src="https://github.com/DeepRegNet/DeepReg/blob/263-Longitudinal-Registration-MICCAI2020/contrib/Longitudinal-Registration-MICCAI2020/figures/pipline.png" width="800"/>


## Getting started
### Installation
- Clone this repo and change to the corresponding branch:
```bash
git clone https://github.com/DeepRegNet/DeepReg.git
cd DeepReg
git checkout 263-Longitudinal-Registration-MICCAI2020
cd ./contrib/Longitudinal-Registration-MICCAI2020
```

### Setting up virtual environment
```bash
conda create -n longi-reg python=3.7 tensorflow-gpu=2.0.0
conda activate longi-reg
pip install -r requirements.txt
```

### Data preparing
- Please store your own data set in `contrib/Longitudinal-Registration-MICCAI2020/data`
- 2 types of files are required before training:
  - A h5 format data file, which contains the processed images with corresponding keys (e.g. `PatientX-VisitN`) as index.
  - A key file, which contains the paired-wise keys of different images, which you're going to register.
  - Fake h5 data files and key files can be generated using `python xxxx.py`, which could be regard as a reference for setting up your own data sets.

### Training
- The command line scripts for reproducing experiments in Table 1 and 2 can be found in `contrib/Longitudinal-Registration-MICCAI2020/scripts`
- Take the experiment of inter and intra patient registration with mmd loss (i.e. IT+IF+IB) as example:
```
bash
cd scripts
bash train_IT+IF+IB.sh
```
- You can use `python main_h5.py --help` to see the definition of the hyper-parameters in the command line.
- To view the loss plot, run `tensorboard --logdir ./logs`

### Prediction

