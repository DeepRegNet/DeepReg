# Unpaired lung CT registration

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/unpaired_ct_lung)

## Author

DeepReg Development Team (Shaheer Saeed)

## Application

This is a registration between CT images from different patients. The images are all
from acquired at the same timepoint in the breathing cycle. This is an inter subject
registration. This kind of registration is useful for determining how one stimulus
affects multiple patients. If a drug or invasive procedure is administered to multiple
patients, registering the images from different patients can give medical professionals
a sense of how each patient is responding in comparison to others. An example of such an
application can be seen in [2].

## Instruction

- [Install DeepReg](https://deepreg.readthedocs.io/en/latest/getting_started/install.html);
- Change current directory to the root directory of DeepReg project;
- The `demo_data.py`, `demo_train.py` and `demo_predict.py` scripts need to be run using
  the following command:

```bash
python3 demos/unpaired_ct_lung/script_name.py
```

A short description of the scripts is provided below. The scripts must be run in the
following order:

- Run the `demo_data.py` script: This script does the following:
  - Download data using `tf.keras.utils.get_file`. Data is downloaded to the demo
    directory but this can be changed (instructions in the comments in the script).
  - Split the data into three sets train, valid and test (change
    ratio_of_test_and_valid_samples variable to change the ratio of test and valid
    samples)
  - Restructure the files, for each of the train, valid and test sets, into a directory
    structure that is suitable for use with the unpaired loader in DeepReg
  - Rescale all images to 0-255 so they are suitable for use with DeepReg
  - Download a pretrained model for use with the predict script
- Run the demo_train.py script: This script does the following:
  - Specify the training options like GPU support
  - Specify the config file paths (the config file to define the network is one which is
    avialable with DeepReg and the config file for the data is given in the demo folder)
  - Train a network using DeepReg
- Run the demo_predict.py script: This script does the following:
  - Use the pretrained network to make predictions for the test set
  - Use the predictions to plot the results (the path to the images generated in the
    logs will need to be specified, look at comments in the script to change this)

## Pre-trained Model

A pre-trained model will be downloaded after running `demo_data.py` and unzipped at the
dataset folder under the demo folder. This pre-trained model will be used by default
with `deepreg_predict`. Run the user-trained model by specifying with `--ckpt_path` the
location where the ckpt files will be saved, in this case (specified by `deepreg_train`
as above), /logs/learn2reg_t2_unpaired_train_logs/.

## Data

The dataset for this demo comes from [1] and can be downloaded from:

https://zenodo.org/record/3835682#.XsUWXsBpFhE

## Tested DeepReg version

Last commit at which demo was tested: c709a46c345552ae1396e6d7ba46a44f7950aea0

Note: This demo was tested using one Nvidia Tesla V100 GPU with a memory of 32GB. Please
ensure that enough memory is available to run the demo otherwise memory allocation
errors might arise.

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose).

## Reference

[1] Hering A, Murphy K, and van Ginneken B. (2020). Lean2Reg Challenge: CT Lung
Registration - Training Data [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3835682

[2] Li B, Christensen GE, Hoffman EA, McLennan G, Reinhardt JM. Establishing a normative
atlas of the human lung: intersubject warping and registration of volumetric CT images.
Acad Radiol. 2003;10(3):255-265. doi:10.1016/s1076-6332(03)80099-5
