# Unpaired ultrasound images - an example for cross validation

## Author

Yipeng Hu (yipeng.hu@ucl.ac.uk)

## Instruction

<!---
"""bash config_generator cross --data_folders dataset/fold0 dataset/fold1 dataset/fold2
dataset/fold3 dataset/fold4 dataset/fold5 dataset/fold6 dataset/fold7 dataset/fold8
dataset/fold9 --prefix unpaired_us_prostate_cv_run
-->

```bash
train --gpu "1, 2, 3" --config_path
demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_run1.yaml
demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_train.yaml --log_dir
demos/unpaired_us_prostate_cv/log """
```

```
predict --gpu "" --config_path
demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_run1.yaml
demos/unpaired_us_prostate_cv/unpaired_us_prostate_cv_train.yaml --log_dir
demos/unpaired_us_prostate_cv/log """
```

## Application

Transrectal ultrasound (TRUS) images are aqcuired from prostate cancer patients.

## Data

The derived unpaired data can be downloaded
https://github.com/ucl-candi/dataset_trus3d/archive/master.zip

## [Tested DeepReg Version]

TBC
