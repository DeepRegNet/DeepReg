# Unpaired lung CT image registration

---

**NOTE**

Please read the
[DeepReg Demo Disclaimer](https://github.com/DeepRegNet/DeepReg/blob/master/demos/README.md).

---

## Author

DeepReg Development Team (raise an issue:
https://github.com/DeepRegNet/DeepReg/issues/new, or mailto the author:
zcemsus@ucl.ac.uk)

## Application

This is a registration between ct images from different patients. The images are all
from acquired at the same timepoint in the breathing cycle. This is an inter subject
registration. This kind of registration is useful for determining how one stimulus
affects multiple patietns. If a drug or invasive procedure is administered to multiple
patients, registering the images from different patients can give medical professsionals
a sense of how each patient is responding in comparison to others. An example of such an
application can be seen in [2].

## Instructions

- [Install DeepReg](https://deepreg.readthedocs.io/en/latest/getting_started/install.html);
- Change current directory to the root directory of DeepReg project;
- The `demo_data.py`, `demo_train.py` and `demo_predict.py` scripts need to be run using
  the following command:

```bash
python3 demos/unpaired_ct_lung/script_name.py
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
    structure that is suitable for use with the unpaired loader in deepreg
  - Rescale all images to 0-255 so they are suitable for use with deepreg
  - Download a pretrained model for use with the predict script
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

The dataset for this demo comes from [1] and can be downloaded from:

https://zenodo.org/record/3835682#.XsUWXsBpFhE

## Tested DeepReg Version

Last commit at which demo was tested: c709a46c345552ae1396e6d7ba46a44f7950aea0

## References

[1] Hering A, Murphy K, and van Ginneken B. (2020). Lean2Reg Challenge: CT Lung
Registration - Training Data [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3835682

[2] Li B, Christensen GE, Hoffman EA, McLennan G, Reinhardt JM. Establishing a normative
atlas of the human lung: intersubject warping and registration of volumetric CT images.
Acad Radiol. 2003;10(3):255-265. doi:10.1016/s1076-6332(03)80099-5
