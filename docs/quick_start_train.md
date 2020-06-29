# Training and Inference

Here is a demo using dummy data set to train a registration network. Read tutorials and
API documentation for more details.

## Training

Train a registration network using unpaired and labeled dummy data with a predefined
configuration:

```bash
train --gpu "" --config_path deepreg/config/unpaired_labeled_ddf.yaml --log_dir test
```

where

- `--gpu ""` means not using GPU. Use `--gpu "0"` to use the GPU of index 0 and use
  `--gpu "0,1"` to use two GPUs.
- `--config_path deepreg/config/unpaired_labeled_ddf.yaml` provides the configuration
  for the training. Read configuration for more details.
- `--log_dir test` specifies the output folder, the output will be saved in `logs/test`.

## Inference

The trained network can be evaluated using unseen dummy test data set:

```bash
predict -g "" --ckpt_path logs/test/save/weights-epoch2.ckpt --mode test
```

where

- `--gpu ""` means not using GPU.
- `--ckpt_path logs/test/save/weights-epoch2.ckpt` provides the checkpoint path of the
  trained network. A copy of training configuration is saved under `logs/test/`, so no
  configuration is required as input.
- `--mode test` means the inference is performed on the test data set. Other options can
  be `train` or `valid`.
