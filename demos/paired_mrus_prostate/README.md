# Paired prostate MR-ultrasound registration

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/paired_mrus_brain)

This demo uses DeepReg to re-implament the algorithms described in
[Weakly-supervised convolutional neural networks for multimodal image registration](https://doi.org/10.1016/j.media.2018.07.002).
A standalone demo was hosted at https://github.com/yipenghu/label-reg.

## Author

DeepReg Development Team

## Application

Registering preoperative MR images to intraoperative transrectal ultrasound images has
been an active research area for more than a decade. The multimodal image registration
task assist a number of ultrasound-guided interventions and surgical procedures, such as
targeted biopsy and focal therapy for prostate cancer patients. One of the key
challenges in this registration task is the lack of robust and effective similarity
measures between the two image types. This demo implements a weakly-supervised learning
approach to learn voxel correspondence between intensity patterns between the multimodal
data, driven by expert-defined anatomical landmarks, such as the prostate gland
segmenation.

## Instruction

- [Install DeepReg](https://deepreg.readthedocs.io/en/latest/getting_started/install.html);
- Change current directory to the root directory of DeepReg project;
- Run `demo_data.py` script to download 10 folds of unpaired 3D ultrasound images and
  the pre-trained model.

```bash
python demos/paired_mrus_prostate/demo_data.py
```

- Run `demo_train` script to launch the training. The following example uses a single
  GPU and launches the first of the ten runs of a 9-fold cross-validation, as specified
  in the [`dataset` section](./paired_mrus_prostate.yaml) and the
  [`train` section](./paired_mrus_prostate.yaml), which can be specified in
  [seperate yaml files](https://deepregnet.github.io/DeepReg/#/tutorial_experiment?id=cross-validation).
  The logs will be saved under `logs_train/` inside the demo folder;

```bash
python demos/paired_mrus_prostate/demo_train.py --no-test
```

- Run `demo_predict` script to use the saved checkpoint file for testing on the data
  partitions specified in the config file, a copy of which will be saved in the log_dir.
  The following example uses a pre-trained model, on CPU. The results will be saved at
  under `logs_predict/` inside the demo folder.

```bash
python demos/paired_mrus_prostate/demo_predict.py
```

## Pre-trained Model

A pre-trained model will be downloaded after running `demo_data.py` and unzipped at the
dataset folder under the demo folder. This pre-trained model will be used by default
with `deepreg_predict`. Run the user-trained model by specifying with `--ckpt_path` the
location where the checkpoint files are saved.

## Data

This is a demo without real clinical data due to regulatory restrictions. The MR and
ultrasound images used are simulated dummy images.

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose).
