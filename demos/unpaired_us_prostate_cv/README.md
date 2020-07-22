# Unpaired ultrasound images - an example for cross validation

## Author

Yipeng Hu (yipeng.hu@ucl.ac.uk)

## Instruction

<!---
"""bash config_generator cross --data_folders dataset/fold0 dataset/fold1 dataset/fold2
dataset/fold3 dataset/fold4 dataset/fold5 dataset/fold6 dataset/fold7 dataset/fold8
dataset/fold9 --prefix unpaired_us_prostate_cv_run
-->

- Change current directory to the root directory of DeepReg project;
- Run `demo_data.py` script to download example 10 folds of unpaired 3D ultrasound
  images;

```bash
python demos/unpaired_us_prostate_cv/demo_data.py
```

- [Install DeepReg](https://deepregnet.github.io/DeepReg/#/quick_start?id=install-the-package)
  first. Then call `train` from command line. The following example uses three GPUs and
  launches the first of the ten runs of a 9-fold cross-validation, as specified in the
  [`dataset` section](./unpaired_us_prostate_cv_run1.yaml) and the
  [`train` section](./unpaired_us_prostate_cv_train.yaml), which can be specified in
  [seperate yaml files](https://deepregnet.github.io/DeepReg/#/tutorial_experiment?id=cross-validation).
  The 10th fold is reserved for testing;

```bash
train --gpu "1, 2, 3" --config_path demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_run1.yaml demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_train.yaml --log_dir demos/unpaired_us_prostate_cv/log
```

- Call `predict` from command line to use the saved ckpt file for testing on the 10th
  fold data. The following example uses a pre-trained model, on CPU.

```bash
predict --gpu "" --config_path demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_run1.yaml --ckpt_path demos/unpaired_us_prostate_cv/log/weights-epoch2.ckpt --mode test --log_dir demos/unpaired_us_prostate_cv/log
```

## Application

Transrectal ultrasound (TRUS) images are aqcuired from prostate cancer patients.
Registering

## Data

The 3D ultrasound images used in this demo were derived from the Prostate-MRI-US-Biopsy
dataset, hosted at the Cancer Imaging Archive (TCIA) at
https://www.cancerimagingarchive.net/.

## [Tested DeepReg Version]

0.14
