# Pairwise Registration for Grouped Prostate Images

This demo uses DeepReg to demonstrate a number of features:

- For grouped data in h5 files, e.g. "group-1-2" indicates the 2th visit from Subject 1;
- Use masks as the images for feature-based registration - aligning the prostate gland
  segmentation in this case - with deep learning;
- Register intra-patient longitudinal data.

## Author

Yipeng Hu

yipeng.hu@ucl.ac.uk

## Application

Logitudinal registration detects the temporal changes and normalises the spatial
difference between images aquired at different time-points. For prostate cancer patients
under active surveillance programmes, quantifying these changes is useful for detecting
and monitoring potential cancerous regions.

## Instruction

- [Install DeepReg](https://deepregnet.github.io/DeepReg/#/quick_start?id=install-the-package);
- Change current directory to the root directory of DeepReg project;
- Run [demo_data.py](./demo_data.py) script to download example 10 folds of unpaired 3D
  ultrasound images and the pre-trained model.

```bash
python demos/grouped_mask_prostate_longitudinal/demo_data.py
```

- Call `deepreg_train` from command line. The following example uses a single GPU and
  launches the first of the ten runs of a 9-fold cross-validation, as specified in the
  [`dataset` section](./grouped_mask_prostate_longitudinal_dataset0.yaml) and the
  [`train` section](./grouped_mask_prostate_longitudinal_train.yaml), which can be
  specified in
  [seperate yaml files](https://deepregnet.github.io/DeepReg/#/tutorial_experiment?id=cross-validation);

```bash
deepreg_train --gpu "0" --config_path demos/grouped_mask_prostate_longitudinal/grouped_mask_prostate_longitudinal.yaml --log_dir grouped_mask_prostate_longitudinal
```

- Call `deepreg_predict` from command line to use the saved ckpt file for testing on the
  data partitions specified in the config file, a copy of which woule be saved in the
  [log_dir]. The following example uses a pre-trained model, on CPU. If not specified,
  the results will be saves at the created timestamp-named directories under /logs.

```bash
deepreg_predict --gpu "" --config_path demos/grouped_mask_prostate_longitudinal/grouped_mask_prostate_longitudinal.yaml --ckpt_path demos/grouped_mask_prostate_longitudinal/dataset/pre-trained/weights-epoch500.ckpt --save_png --mode test
```

## Pre-trained Model

A pre-trained model will be downloaded after running [demo_data.py](./demo_data.py) and
unzipped at dataset folder under the demo folder. This pre-trained model will be used by
default with `deepreg_predict`. Run the user-trained model by specify `--ckpt_path` to
where the ckpt files are save, in this case (specified by `deepreg_train` as above),
/logs/grouped_mask_prostate_longitudinal/.

## Data

This is a demo without real clinical data due to regulatory restrictions. The MR and
ultrasound images used are simulated dummy images.

## Tested DeepReg Version

Last commit at which demo was tested: 3157f880eb99ce10fc3a4a8ebcc595bd67be24e1
