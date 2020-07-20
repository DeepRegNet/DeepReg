# Paired CT Lung Registration

## Author

Shaheer U. Saeed

zcemsus@ucl.ac.uk

## Instructions

- Run the demo_data.py script: This script does the following:
  - Download data using linux builtin function wget (if using other OS please use python
    package wget, instruction in the script). Data is downloaded to the demo directory
    but this can be changed (instructions in the comments in the script).
  - Split the data into three sets train, valid and test (change
    ratio_of_test_and_valid_samples variable to change the ratio of test and valid
    samples)
  - Restructure the files, for each of the train, valid and test sets, into a directory
    structure that is suitable for use with the paired loader in deepreg
  - Rescale all images to 0-255 so they are suitable for use with deepreg
- Run the demo_train.py script: This script does the following:
  - Specify the training options like gpu support
  - Specify the config file paths (the config file to define the network is one which is
    avialable with deepreg and the config file for the data is given in the demo folder)
  - Train a network using deepreg
- Run the demo_predict.py script: This script does the following:
  - Use the trained network to make predictions for the test set
  - Use the predicitons to plot the results (the path to the images generated in the
    logs will need to be sepcified, look at comments in the script to chnage this)

## Application

This is a registration between ct images acquired at different time points for a single
patient. The images being registered are taken at inspiration and expiration for each
subject. This is an intra subject registration. This type of intra subject registration
is useful when there is a need to track certain features on a medical image such as
tumor location when conducting invasive procedures.

## Data

The dataset for this demo comes from [1] and can be downloaded from:

https://zenodo.org/record/3835682#.XsUWXsBpFhE

## Tested DeepReg Version

v0.1.4

## References

[1] Hering, Alessa, Murphy,Keelin, and van Ginneken, Bram. (2020). Lean2Reg Challenge:
CT Lung Registration - Training Data [Data set]. Zenodo.
http://doi.org/10.5281/zenodo.3835682
