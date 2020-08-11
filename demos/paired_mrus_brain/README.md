# Paired MR US Brain Registration

## Author

Shaheer U. Saeed

zcemsus@ucl.ac.uk

## Application

This is a registration between MR and Ultrasound images of the brain. The registration
is a paired unlabeled intra-subject registration. This type of registration is useful in
numerous scenarios, one example is using imaging techniques that can be used
intra-operatively like ultrasound to localise tumors using pre-operative images acquired
from imaging such as MR.

## Instructions

- [Install DeepReg](https://deepregnet.github.io/DeepReg/#/quick_start?id=install-the-package);
- Change current directory to the root directory of DeepReg project;
- The `demo_data.py`, `demo_train.py` and `demo_predict.py` scripts need to be run using
  the following command:

```bash
python3 demos/paired_mrus_brain/script_name.py
```

A short description of the scripts is provided below. The scripts must be run in the
following order:

- Run the demo_data.py script: This script does the following:
  - Download the data by cloning a github repository. The repository contians only a
    reduced dataset which has been preprocessed. The script also has code to download
    and preprocess the full dataset. The code block titled "FULL DATA DOWNLOAD AND
    PREPROCESS" must be uncommented and the code block titled "PARTIAL PREPROCESSED DATA
    DOWNLOAD" must be commented out in order to download and use the full data (further
    instructions provided in the script). To use the full data please edit eh config
    file to specify the moving_image_shape as [256, 256, 288] as well.
  - Split the data into train and test sets (change ratio_of_test variable to change the
    ratio of test samples)
  - Restructure the files, for each of the train and test sets, into a directory
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
- Note: The number of epochs and reduced dataset size for training will result in a loss
  in test accuracy so please train with the full dataset and for a greater number of
  epochs for improved results.

## Data

The dataset for this demo comes from [1] and can be downloaded from:

https://archive.sigma2.no/pages/public/datasetDetail.jsf?id=10.11582/2020.00025

## Tested DeepReg Version

Last commit at which demo was tested: bee29caf4f63e89cca7f41e8e664454670f2f763

## References

[1] Y. Xiao, M. Fortin, G. Unsgård , H. Rivaz, and I. Reinertsen, “REtroSpective
Evaluation of Cerebral Tumors (RESECT): a clinical database of pre-operative MRI and
intra-operative ultrasound in low-grade glioma surgeries”. Medical Physics, Vol. 44(7),
pp. 3875-3882, 2017.
