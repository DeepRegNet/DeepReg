# Paired MR-to-Ultrasound registration - an example of weakly-supervised label-driven training

This demo uses DeepReg to re-implament the algorithms described in
[Weakly-supervised convolutional neural networks for multimodal image registration](https://doi.org/10.1016/j.media.2018.07.002).
A standalone demo was hosted at https://github.com/YipengHu/label-reg.

## Author

Yipeng Hu (yipeng.hu@ucl.ac.uk)

## Application

Registering preoperative MR images to intraoperative transrectal ultrasound images has
been an active research area for more than a decade. The multimodal image registration
task assist a number of ultrasound-guided intervention and surgical procedures, such as
targted biopsy and focal therapy for prostate cancer patients. One of the key challenges
in this registration tasks is the lack of robust and effective similarity measures
between the two image types. This demo implements a weakly-supervised learning approach
to learn voxel correspondence between intensity patterns between the multimodal data,
driven by expert-defined anatomical landmarks, such as the prostate gland segmenation.

## Instruction

- [Install DeepReg](https://deepregnet.github.io/DeepReg/#/quick_start?id=install-the-package);
- Change current directory to the root directory of DeepReg project;
- Run [demo_data.py](./demo_data.py) script to download example 10 folds of unpaired 3D
  ultrasound images and the pre-trained model.

```bash
python demos/paired_mrus_prostate/demo_data.py
```

- Call `deepreg_train` from command line. The following example uses two GPUs and
  launches the first of the ten runs of a 9-fold cross-validation, as specified in the
  [`dataset` section](./paired_mrus_prostate_dataset0.yaml) and the
  [`train` section](./paired_mrus_prostate_train.yaml), which can be specified in
  [seperate yaml files](https://deepregnet.github.io/DeepReg/#/tutorial_experiment?id=cross-validation);

```bash
deepreg_train --gpu "1, 2" --config_path demos/paired_mrus_prostate/paired_mrus_prostate_dataset0.yaml demos/paired_mrus_prostate/paired_mrus_prostate_train.yaml --log_dir paired_mrus_prostate
```

- Call `deepreg_predict` from command line to use the saved ckpt file for testing on the
  data partitions specified in the config file, a copy of which woule be saved in the
  [log_dir]. The following example uses a pre-trained model, on CPU. If not specified,
  the results will be saves at the created timestamp-named directories under /logs.

```bash
deepreg_predict --gpu "" --config_path demos/paired_mrus_prostate/paired_mrus_prostate_dataset0.yaml demos/paired_mrus_prostate/paired_mrus_prostate_train.yaml --ckpt_path demos/paired_mrus_prostate/dataset/pre-trained/weights-epoch500.ckpt --mode test
```

## Pre-trained Model

A pre-trained model will be downloaded after running [demo_data.py](./demo_data.py) and
unzipped at dataset folder under the demo folder. This pre-trained model will be used by
default with `deepreg_predict`. Run the user-trained model by specify `--ckpt_path` to
where the ckpt files are save, in this case (specified by `deepreg_train` as above),
/logs/paired_mrus_prostate/.

## Data

This is a demo without real clinical data due to regulatory restrictions. The MR and
ultrasound images used are simulated dummy images.

## Tested DeepReg Tag

#7ec0f5157a81cd5e60cadd61bd617b433039d0e6
