# Paired MR and US brain image registration

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

## Author

DeepReg Development Team (raise an issue:
https://github.com/DeepRegNet/DeepReg/issues/new, or mailto the author:
zcemsus@ucl.ac.uk)

## Application

This is a registration between MR and Ultrasound images of the brain. The registration
is a paired unlabeled intra-subject registration. This type of registration is useful in
numerous scenarios, one example is using imaging techniques that can be used
intra-operatively like ultrasound to localise tumors using pre-operative images acquired
from imaging such as MR.

## Instructions

- [Install DeepReg](https://deepreg.readthedocs.io/en/latest/getting_started/install.html);
- Change current directory to the root directory of DeepReg project;
- The `demo_data.py`, `demo_train.py` and `demo_predict.py` scripts need to be run using
  the following command:

```bash
python3 demos/paired_mrus_brain/script_name.py
```

A short description of the scripts is provided below. The scripts must be run in the
following order:

- Run the demo_data.py script: This script does the following:
  - Download a reuced copy of the dataset which has already been preprocessed
  - Dowload a pretrained model for use with the predict function
  - Note: This script can also be used to work with the full dataset by uncommenting the
    relevant sections in the script (please read comments in scripts to see what to
    comment out and what to uncomment to use the full dataset)
- Run the demo_train.py script: This script does the following:
  - Specify the training options like gpu support
  - Specify the config file paths (the config file to define the network is one which is
    avialable with deepreg and the config file for the data is given in the demo folder)
  - Train a network using deepreg
- Run the demo_predict.py script: This script does the following:
  - Use the pretrained network to make predictions for the test set
  - Use the predicitons to plot the results (the path to the images generated in the
    logs will need to be sepcified, look at comments in the script to chnage this)
- Note: The number of epochs and reduced dataset size for training will result in a loss
  in test accuracy so please train with the full dataset and for a greater number of
  epochs for improved results.

## Data

The dataset for this demo comes from Xiao et al. [1] and can be downloaded from:

https://archive.sigma2.no/pages/public/datasetDetail.jsf?id=10.11582/2020.00025

## Tested DeepReg Version

Last commit at which demo was tested: c709a46c345552ae1396e6d7ba46a44f7950aea0

## References

[1] Y. Xiao, M. Fortin, G. Unsgård , H. Rivaz, and I. Reinertsen, "REtroSpective
Evaluation of Cerebral Tumors (RESECT): a clinical database of pre-operative MRI and
intra-operative ultrasound in low-grade glioma surgeries". Medical Physics, Vol. 44(7),
pp. 3875-3882, 2017.
