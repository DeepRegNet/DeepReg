# Paired lung CT image registration

## Author

DeepReg Development Team (raise an issue:
https://github.com/DeepRegNet/DeepReg/issues/new, or mailto the author:
zcemsus@ucl.ac.uk)

## Application

This is a registration between ct images acquired at different time points for a single
patient. The images being registered are taken at inspiration and expiration for each
subject. This is an intra subject registration. This type of intra subject registration
is useful when there is a need to track certain features on a medical image such as
tumor location when conducting invasive procedures.

## Instructions

- [Install DeepReg](https://deepregnet.github.io/DeepReg/#/quick_start?id=install-the-package);
- Change current directory to the root directory of DeepReg project;
- The `demo_data.py`, `demo_train.py` and `demo_predict.py` scripts need to be run using
  the following command:

```bash
python3 demos/paired_ct_lung/script_name.py
```

A short description of the scripts is provided below. The scripts must be run in the
following order:

- Run the demo_data.py script: This script does the following:
  - Download data using `tf.keras.utils.get_file`. Data is downloaded to the demo
    directory but this can be changed (instructions in the comments in the script).
  - Split the data into three sets train, valid and test (change
    ratio_of_test_and_valid_samples variable to change the ratio of test and valid
    samples)
  - Restructure the files, for each of the train, valid and test sets, into a directory
    structure that is suitable for use with the paired loader in deepreg
  - Rescale all images to 0-255 so they are suitable for use with deepreg
  - Download a pretrained model to use with the predict script
- Run the demo_train.py script: This script does the following:
  - Specify the training options like gpu support
  - Specify the config file paths (the config file to define the network is one which is
    avialable with deepreg and the config file for the data is given in the demo folder)
  - Train a network using deepreg
- Run the demo_predict.py script: This script does the following:
  - Use the pretrained network to make predictions for the test set
  - Use the predicitons to plot the results (the path to the images generated in the
    logs will need to be sepcified, look at comments in the script to chnage this)

## Data

The dataset for this demo comes from
[Lean2Reg Challenge: CT Lung Registration - Training Data](https://zenodo.org/record/3835682#.XsUWXsBpFhE)
[1].

## Tested DeepReg Version

Last commit at which demo was tested: c709a46c345552ae1396e6d7ba46a44f7950aea0

## References

[1] Hering, Alessa, Murphy,Keelin, and van Ginneken, Bram. (2020). Lean2Reg Challenge:
CT Lung Registration : CT Lung Registration - Training Data
