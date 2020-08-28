### Official inplementation for MICCAI 2020 paper: 
### Longitudinal image registration with temporal-order and subject-specificity discrimination

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
  - A h5 format data file, which contains the processed images with corresponding keys (e.g. `PatientX-VisitN`) as index. For longitudinal MRI data, each patient could have serveral follow-up visits in a period of time.The `X` stands for the patient ID and N stands for the time point of the visits. 
  - A key file, which contains the paired-wise keys of different images which you are going to register. For example, key pair `(Patient1-Visit1, Patient1-Visit2)` means registering images accquired at different visits from the same patient, a typical sample for intra-patient registration. `(Patient1-Visit1, Patient2-Visit1)` means registering images from different patients -- an sample for inter-patient registration.
  - Fake h5 data files and key files can be generated using `python xxxx.py`, which could be regard as a reference for setting up your own data sets.

### Training
- The command line scripts for reproducing experiments in Table 1 and 2 can be found in `contrib/Longitudinal-Registration-MICCAI2020/scripts`
- Take the experiment of inter and intra patient registration with mmd loss (i.e. IT+IF+IB_mmd) as example:
```
bash
cd scripts
bash train_IT+IF+IB_mmd.sh
```
- You can use `python main_h5.py --help` to see the definition of the hyper-parameters in the command line.
- To view the loss plot, run `tensorboard --logdir ./logs`

### Prediction
 - The bash script for prediction is `./scripts/test.sh`, the explaination for the parameters for prediction are listed as follows:
   - `exp_name`: The name of the experiment that you want to do prediction. It should be the same as the `exp_name` in its corresponding train bash scripts. If you have run a train scripts multiple times with the same experiment name, the code will add a time stamp to the name in order to discriminate from each other.  In this case, you should add the corresponding time stame to the original experiment name, as listed in the `./logs` folder.
   - `data_file` and `key_file`: The same h5 data and pkl key files you used in training.
   - `test_phase`: `--test_phase test` will make prediction and calculate evaluation metrics on validation set (for tunning hyper-parameters, usually combined with `--test_model_start_end` to predict with multiple checkpoints, see below). `--test_phase holdout` will make prediction on the holdout test set (only used once for report, see below for details).
   - `test_model_start_end`: If doing prediction on validation set (i.e. using `--test_phase test`), inference and evaluation should be done on multiple checkpoints. If an experiment have dumped 80 checkpoints in training, you can set `--test_model_start_end 60 80` to just evaluates last 20 checkpoints that been saved. The test script will print the checkpoints which got the best on each evaluation metric.
   - `continue_epoch`: If doing prediction on holdout test set (i.e. using `--test_phase holdout`), this param will specify the only checkpoint that you want to use for inference. E.g. `--continue_epoch 299`. This params could also be used in the training scripts for restoring any experiments.
   - `test_gen_pred_imgs`: If set 1, the images and segmentations for before and after registration will be saved in the corresponding experiment folder in the `./logs`. Just for visualization purposes.
<img src="https://github.com/DeepRegNet/DeepReg/blob/263-Longitudinal-Registration-MICCAI2020/contrib/Longitudinal-Registration-MICCAI2020/figures/vis.png" width="800"/>
