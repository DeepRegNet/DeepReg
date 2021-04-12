# Quick Start

This is a set of simple tests to use DeepReg command line tools. More details and other
options can be found in [Command Line Tools](../docs/cli.html).

First, [install DeepReg](install.html) and change current directory to the root
directory of DeepReg.

## Train a registration network

Train a registration network using unpaired and labeled example data with a predefined
configuration:

```bash
deepreg_train --gpu "" --config_path config/unpaired_labeled_ddf.yaml --exp_name test
```

where:

- `--gpu ""` indicates using CPU. Change to `--gpu "0"` to use the GPU at index 0.
- `--config_path <filepath>` specifies the configuration file path.
- `--log_dir test` specifies the output folder. In this case, the output is saved in
  `logs/test`.

## Evaluate a trained network

Once trained, evaluate the network using a test dataset:

```bash
deepreg_predict --gpu "" --ckpt_path logs/test/save/ckpt-2 --split test
```

where:

- `--ckpt_path <filepath>` specifies the checkpoint file path.
- `--split test` specifies prediction on the test dataset.

## Warp an image

DeepReg provides a command line interface (CLI) tool to warp an image/label with a dense
displacement field (DDF):

```bash
deepreg_warp --image data/test/nifti/unit_test/moving_image.nii.gz --ddf data/test/nifti/unit_test/ddf.nii.gz --out logs/test_warp/out.nii.gz
```

where:

- `--image <filepath>` specifies the image/label file path.
- `--ddf <filepath>` specifies the ddf file path.
- `--out <filepath>` specifies the output file path.
