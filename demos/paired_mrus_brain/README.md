# Paired MR and US brain image registration

---

**NOTE**

Please read the
[DeepReg Demo Disclaimer](https://deepreg.readthedocs.io/en/325-improve-contributing-pages/demo/introduction.html#demo-disclaimer).

---

## Author

DeepReg Development Team (raise an issue:
https://github.com/DeepRegNet/DeepReg/issues/new, or mail to the author:
zcemsus@ucl.ac.uk)

## Application

This demo aims to register pairs of brain MR and ultrasound scans. The dataset consists
of 22 subjects with low-grade brain gliomas who underwent brain tumour resection [1].
The main application for this type of registration is to better delineate brain tumour
boundaries during surgery and correct tissue shift induced by the craniotomy.

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
  - Download a reduced copy of the dataset which has already been preprocessed
  - Download a pretrained model for use with the predict function
  - Note: This script can also be used to work with the full dataset by uncommenting the
    relevant sections in the script (please read the scripts' comments to see how to
    download the full dataset)
- Run the demo_train.py script: This script does the following:
  - Specify the training options like gpu support
  - Specify the config file paths (to define both the network config available in
    deepreg and the data config given in the demo folder)
  - Train a network using deepreg
- Run the demo_predict.py script: This script does the following:
  - Use the pretrained network to make predictions for the test set
  - Use the predicitions to plot the results (the images path generated in the logs will
    need to be specified)
- Note: The number of epochs and reduced dataset size for training will result in a loss
  in test accuracy so please train with the full dataset and for a greater number of
  epochs for improved results.

## Data

The dataset for this demo comes from Xiao et al. [1] and can be downloaded from:

https://archive.sigma2.no/pages/public/datasetDetail.jsf?id=10.11582/2020.00025

## Tested DeepReg Version

Last commit at which demo was tested: c709a46c345552ae1396e6d7ba46a44f7950aea0

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new) following the
[guidelines](https://deepreg.readthedocs.io/en/325-improve-contributing-pages/contributing/issue.html).

## References

[1] Y. Xiao, M. Fortin, G. Unsgård , H. Rivaz, and I. Reinertsen, "REtroSpective
Evaluation of Cerebral Tumors (RESECT): a clinical database of pre-operative MRI and
intra-operative ultrasound in low-grade glioma surgeries". Medical Physics, Vol. 44(7),
pp. 3875-3882, 2017.
