# Unpaired prostate ultrasound registration

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/unpaired_us_prostate_cv)

This DeepReg Demo is also an example of cross validation.

## Author

DeepReg Development Team

## Application

Transrectal ultrasound (TRUS) images are acquired from prostate cancer patients during
image-guided procedures. Pairwise registration between these 3D images may be useful for
intraoperative motion modelling and group-wise registration for population studies.

## Data

The 3D ultrasound images used in this demo were derived from the Prostate-MRI-US-Biopsy
dataset, hosted at the
[Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/).

## Instruction

<!---
"""bash config_generator cross --data_folders dataset/fold0 dataset/fold1 dataset/fold2
dataset/fold3 dataset/fold4 dataset/fold5 dataset/fold6 dataset/fold7 dataset/fold8
dataset/fold9 --prefix unpaired_us_prostate_cv_run
-->

- [Install DeepReg](https://deepreg.readthedocs.io/en/latest/getting_started/install.html);
- Change current directory to the root directory of DeepReg project;
- Run `demo_data.py` script to download 10 folds of unpaired 3D ultrasound images;

```bash
python demos/unpaired_us_prostate_cv/demo_data.py
```

- Call `deepreg_train` from command line. The following example uses three GPUs and
  launches the first of the ten runs of a 9-fold cross-validation, as specified in the
  [`dataset` section](./unpaired_us_prostate_cv_run1.yaml) and the
  [`train` section](./unpaired_us_prostate_cv_train.yaml), which can be specified in
  [separate yaml files](https://deepreg.readthedocs.io/en/latest/tutorial/cross_val.html).
  The 10th fold is reserved for testing;

```bash
deepreg_train --gpu "1, 2, 3" --config_path demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_run1.yaml demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_train.yaml --log_dir unpaired_us_prostate_cv
```

- Call `deepreg_predict` from command line to use the saved ckpt file for testing on the
  10th fold data. The following example uses a pre-trained model, on CPU. If not
  specified, the results will be saved at the created timestamp-named directories under
  /logs.

```bash
deepreg_predict --gpu "" --config_path demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_run1.yaml demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_train.yaml --ckpt_path demos/unpaired_us_prostate_cv/dataset/pre-trained/weights-epoch5000.ckpt --mode test
```

## Pre-trained Model

A pre-trained model will be downloaded after running `demo_data.py` and unzipped at the
dataset folder under the demo folder. This pre-trained model will be used by default
with `deepreg_predict`. Run the user-trained model by specifying with `--ckpt_path` the
location where the ckpt files will be saved, in this case (specified by `deepreg_train`
as above), /logs/unpaired_us_prostate_cv/.

## Tested DeepReg version

Last commit at which demo was tested: 7bec018b5e910f1589888f3f286e9f6a11060c31

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose).
