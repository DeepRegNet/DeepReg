# Paired brain MR-ultrasound registration

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/paired_mrus_brain)

## Author

DeepReg Development Team (Shaheer Saeed)

## Application

This demo aims to register pairs of brain MR and ultrasound scans. The dataset consists
of 22 subjects with low-grade brain gliomas who underwent brain tumour resection [1].
The main application for this type of registration is to better delineate brain tumour
boundaries during surgery and correct tissue shift induced by the craniotomy.

## Instruction

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
  - Specify the training options like GPU support
  - Specify the config file paths (to define both the network config available in
    DeepReg and the data config given in the demo folder)
  - Train a network using DeepReg
- Run the demo_predict.py script: This script does the following:
  - Use the pretrained network to make predictions for the test set
  - Use the predicitions to plot the results (the images path generated in the logs will
    need to be specified)
- Note: The number of epochs and reduced dataset size for training will result in a loss
  in test accuracy so please train with the full dataset and for a greater number of
  epochs for improved results.

## Pre-trained Model

A pre-trained model will be downloaded after running `demo_data.py` and unzipped at the
dataset folder under the demo folder. This pre-trained model will be used by default
with `deepreg_predict`. Run the user-trained model by specifying with `--ckpt_path` the
location where the ckpt files will be saved, in this case (specified by `deepreg_train`
as above), /logs/learn2reg_t1_paired_train_logs/.

## Data

The dataset for this demo comes from Xiao et al. [1] and can be downloaded from:

https://archive.sigma2.no/pages/public/datasetDetail.jsf?id=10.11582/2020.00025

## Tested DeepReg version

Last commit at which demo was tested: c709a46c345552ae1396e6d7ba46a44f7950aea0

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose).

## Reference

[1] Y. Xiao, M. Fortin, G. Unsgård , H. Rivaz, and I. Reinertsen, "REtroSpective
Evaluation of Cerebral Tumors (RESECT): a clinical database of pre-operative MRI and
intra-operative ultrasound in low-grade glioma surgeries". Medical Physics, Vol. 44(7),
pp. 3875-3882, 2017.
