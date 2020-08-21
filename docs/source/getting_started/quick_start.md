# Quick Start

Following is a simplified demo using example dataset to train a registration network.
More details and other options can be found in the [command line tools](doc_command.md).

## Train a registration network

Train a registration network using unpaired and labeled example data with a predefined
configuration:

```bash
deepreg_train --gpu "" --config_path deepreg/config/unpaired_labeled_ddf.yaml --log_dir test
```

where

- `--gpu ""` indicates using CPU. Change to `--gpu "0"` to use the GPU of index 0.
- `--config_path deepreg/config/unpaired_labeled_ddf.yaml` specifies the configuration
  file path.
- `--log_dir test` specifies the output folder, the output will be saved in `logs/test`.

## Evaluate a trained network

The trained network can be evaluated using unseen example test dataset:

```bash
deepreg_predict --gpu "" --ckpt_path logs/test/save/weights-epoch2.ckpt --mode test
```

where

- `--ckpt_path logs/test/save/weights-epoch2.ckpt` specifies the checkpoint file path.
- `--mode test` specifies prediction on the test dataset.

## Warp an image

DeepReg provides a command line interface (CLI) tool to warp an image / label with a
dense displacement field (DDF):

```bash
deepreg_warp --image data/test/nifti/unit_test/moving_image.nii.gz --ddf data/test/nifti/unit_test/ddf.nii.gz --out logs/test_warp/out.nii.gz
```

where

- `--image` provides the file path of the image/label.
- `--ddf` provides the file path of the ddf.
- `--out`, provides the file path for the output.
