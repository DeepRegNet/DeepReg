# DeepReg Demo README Example

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/)

## Author

DeepReg Development Team (optional corresponding Author Name) or Author Name for
external contributor.

## Application

Briefly describe the clinical application and the need for registration.

## Data

Describe the source of the dataset. Dataset should be open-accessible.

## Instruction

### Installation

Please install DeepReg following the [instructions](../getting_started/install.html) and
change the current directory to the root directory of DeepReg project, i.e. `DeepReg/`.

### Download data

Please execute the following command to download the data and pre-trained model.

```bash
python demos/name/demo_data.py
```

### Launch demo training

Please execute the following command to launch a demo training. The training logs will
be saved under `demos/name/logs_train`, where the saved checkpoints can be used for
prediction later.

```bash
python demos/name/demo_train.py
```

Here the training is launched using the GPU of index 0 with a limited number of steps
and reduced size. Please add flag `--no-test` to use the original training
configuration, such as

```bash
python demos/name/demo_train.py --no-test
```

### Launch prediction

Please execute the following command to launch the prediction with pre-trained model.
The prediction logs will be saved under `demos/name/logs_predict`, where the
visualization of predictions are saved. Check the [CLI documentation](../docs/cli.html)
for more details about prediction output.

```bash
python demos/name/demo_predict.py
```

Optionally, the user-trained model can be used by changing the `ckpt_path` variable
inside `demo_predict.py`. Note that the path should end with `.ckpt` and checkpoints are
saved under `logs_train` as mentioned above.

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose) for any
questions.

## Reference

Related references in the form of `[1] Reference`.
