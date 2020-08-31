# Pairwise registration for grouped prostate segmentation masks

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/grouped_mask_prostate_longitudinal)

This demo uses DeepReg to demonstrate a number of features:

- For grouped data in h5 files, e.g. "group-1-2" indicates the 2th visit from Subject 1;
- Use masks as the images for feature-based registration - aligning the prostate gland
  segmentation in this case - with deep learning;
- Register intra-patient longitudinal data.

## Author

DeepReg Development Team

## Application

Longitudinal registration detects the temporal changes and normalises the spatial
difference between images acquired at different time-points. For prostate cancer
patients under active surveillance programmes, quantifying these changes is useful for
detecting and monitoring potential cancerous regions.

## Data

This is a demo without real clinical data due to regulatory restrictions. The MR and
ultrasound images used are simulated dummy images.

## Instruction

- [Install DeepReg](https://deepreg.readthedocs.io/en/latest/getting_started/install.html);
- Change current directory to the root directory of DeepReg project;
- Run `demo_data.py` script to download 10 folds of unpaired 3D ultrasound images and
  the pre-trained model.

```bash
python demos/grouped_mask_prostate_longitudinal/demo_data.py
```

- Call `deepreg_train` from command line. The following example uses a single GPU and
  launches the first of the ten runs of a 9-fold cross-validation, as specified in the
  [`dataset` section](./grouped_mask_prostate_longitudinal_dataset0.yaml) and the
  [`train` section](./grouped_mask_prostate_longitudinal_train.yaml), which can be
  specified in
  [separate yaml files](https://deepreg.readthedocs.io/en/latest/tutorial/cross_val.html);

```bash
deepreg_train --gpu "0" --config_path demos/grouped_mask_prostate_longitudinal/grouped_mask_prostate_longitudinal.yaml --log_dir grouped_mask_prostate_longitudinal
```

- Call `deepreg_predict` from command line to use the saved ckpt file for testing on the
  data partitions specified in the config file, a copy of which would be saved in the
  [log_dir]. The following example uses a pre-trained model, on CPU. If not specified,
  the results will be saved at the created timestamp-named directories under /logs.

```bash
deepreg_predict --gpu "" --config_path demos/grouped_mask_prostate_longitudinal/grouped_mask_prostate_longitudinal.yaml --ckpt_path demos/grouped_mask_prostate_longitudinal/dataset/pre-trained/weights-epoch500.ckpt --save_png --mode test
```

## Pre-trained model

A pre-trained model will be downloaded after running `demo_data.py` and unzipped at the
dataset folder under the demo folder. This pre-trained model will be used by default
with `deepreg_predict`. Run the user-trained model by specifying with `--ckpt_path` the
location where the ckpt files will be saved, in this case (specified by `deepreg_train`
as above), /logs/grouped_mask_prostate_longitudinal/.

## Tested DeepReg version

Last commit at which demo was tested: 3157f880eb99ce10fc3a4a8ebcc595bd67be24e1

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose).
