# Paired prostate MR-ultrasound registration

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/paired_mrus_brain)

This demo uses DeepReg to re-implement the algorithms described in
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
segmentaion.

## Data

This is a demo without real clinical data due to regulatory restrictions. The MR and
ultrasound images used are simulated dummy images.

## Instruction

### Installation

Please install DeepReg following the [instructions](../getting_started/install.html) and
change the current directory to the root directory of DeepReg project, i.e. `DeepReg/`.

### Download data

Please execute the following command to download and pre-process the data and
pre-trained model.

```bash
python demos/paired_mrus_prostate/demo_data.py
```

### Launch demo training

Please execute the following command to launch a demo training (the first of the ten
runs of a 9-fold cross-validation). The training logs and model checkpoints will be
saved under `demos/paired_mrus_prostate/logs_train`.

```bash
python demos/paired_mrus_prostate/demo_train.py
```

Here the training is launched using the GPU of index 0 with a limited number of steps
and reduced size. Please add flag `--no-test` to use the original training
configuration, such as

```bash
python demos/paired_mrus_prostate/demo_train.py --no-test
```

### Launch prediction

Please execute the following command to launch the prediction with pre-trained model.
The prediction logs and visualization results will be saved under
`demos/paired_mrus_prostate/logs_predict`. Check the
[CLI documentation](../docs/cli.html) for more details about prediction output.

```bash
python demos/paired_mrus_prostate/demo_predict.py
```

Optionally, the user-trained model can be used by changing the `ckpt_path` variable
inside `demo_predict.py`. Note that the path should end with `.ckpt` and checkpoints are
saved under `logs_train` as mentioned above.

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose) for any
questions.
