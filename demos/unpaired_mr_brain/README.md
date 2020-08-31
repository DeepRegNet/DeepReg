# Unpaired hippocampus MR registration

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/unpaired_mr_brain)

## Author

DeepReg Development Team (Adrià Casamitjana)

## Application

This is a demo targeting the alignment of hippocampal substructures (head and body)
using mono-modal MR images between different patients. The images are cropped around
those areas and manually annotated. This is a 3D intra-modal registration using a
composite loss of image and label similarity.

## Instruction

- [Install DeepReg](https://deepreg.readthedocs.io/en/latest/getting_started/install.html);
- Change current directory to the root directory of DeepReg project;
- The `demo_data.py`, `demo_train.py` and `demo_predict.py` scripts need to be run using
  the following command:

```bash
python3 demos/unpaired_mr_brain/script_name.py
```

A short description of the scripts is provided below. The scripts must be run in the
following order:

- (Optional) Create a new configuration file following the template in
  demos/unpaired_mr_brain/unpaired_mr_brain.yaml. It specifies:
  - Dataset options: input data directory, loader type, data format
  - Model options: backbone network, field type.
  - Training options: losses, optimizer, number of epochs.
- Run the `demo_data.py` script: This script does the following:
  - Download and extract the dataset. Data is downloaded to the demo directory under
    data/ but this can be changed (instructions in the comments in the script).
  - Split subjects into train/test according to the challenge website.
  - Rescale all images to 0-255 so they are suitable for use with DeepReg
  - Create and apply a binary mask to mask-out the padded values in images.
  - Transform label volumes using one-hot encoding (only for foreground classes)
- Run the `demo_train.py` script: This script does the following:
  - Specify the training options like GPU support
  - Specify the config file paths
  - Train a network using DeepReg
- Run the `demo_predict.py` script: This script does the following:
  - Use the trained network to make predictions for the test set
  - Use the predictions to plot the results (the path to the images generated in the
    logs will need to be specified, look at comments in the script to change this)

## Pre-trained Model

A pre-trained model will be downloaded after running `demo_data.py` and unzipped at the
dataset folder under the demo folder. This pre-trained model will be used by default
with `deepreg_predict`. Run the user-trained model by specifying with `--ckpt_path` the
location where the ckpt files will be saved, in this case (specified by `deepreg_train`
as above), /logs/learn2reg_t4_unpaired_train_logs/.

## Data

The dataset for this demo comes from the Learn2Reg MICCAI Challenge (Task 4) [1] and can
be downloaded from:

https://drive.google.com/uc?export=download&id=1RvJIjG2loU8uGkWzUuGjqVcGQW2RzNYA

## Tested DeepReg version

Last commit at which demo was tested: v. 0.1.6-alpha

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose).

## Reference

[1] AL Simpson et al., _A large annotated medical image dataset for the development and
evaluation of segmentation algorithms_ (2019). https://arxiv.org/abs/1902.09063
