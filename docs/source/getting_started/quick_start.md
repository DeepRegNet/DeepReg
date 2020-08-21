# Quick Start

## Train a registration network

Train a registration network using unpaired and labeled example data with a predefined
configuration:

```bash
deepreg_train --gpu "" --config_path deepreg/config/unpaired_labeled_ddf.yaml --log_dir test
```

where

- `--gpu ""` indicates using CPU. `--gpu "0"` uses the GPU of index 0 and `--gpu "0,1"`
  uses two GPUs.
- `--config_path deepreg/config/unpaired_labeled_ddf.yaml` provides the configuration
  for the training. Read configuration for more details.
- `--log_dir test` specifies the output folder, the output will be saved in `logs/test`.

## Predict a displacement field

The trained network can be evaluated using unseen example test dataset:

```bash
deepreg_predict --gpu "" --ckpt_path logs/test/save/weights-epoch2.ckpt --mode test
```

where

- `--gpu ""` indicates using CPU for inference.
- `--ckpt_path logs/test/save/weights-epoch2.ckpt` provides the checkpoint path of the
  trained network. As a copy of training configuration is saved under `logs/test/`
  during training, so no configuration is required as input in this case.
- `--mode test` indicates the inference on the test dataset. Other options include
  `train` or `valid`.

This is a simplified demo using example dataset to train a registration network. More
details and other options can be found in the [command line tools](doc_command.md).

## Warp an image

DeepReg provides a command line interface (CLI) tool to warp an image / label with a
dense displacement field (DDF):

```bash
deepreg_warp --image data/test/nifti/unit_test/moving_image.nii.gz --ddf data/test/nifti/unit_test/ddf.nii.gz --out logs/test_warp/out.nii.gz
```

where

- `--image` provides the file path of the image/label. The image/label should be saved
  in a nifti file with suffix `.nii` or `.nii.gz`. The image/label should be a 3D / 4D
  tensor, where the first three dimensions correspond to the moving image shape and the
  fourth can be a channel of features.
- `--ddf` provides the file path of the ddf. The ddf should be saved in a nifti file
  with suffix `.nii` or `.nii.gz`. The ddf should be a 4D tensor, where the first three
  dimensions correspond to the fixed image shape and the fourth dimension has 3 channels
  corresponding to x, y, z axises.
- `--out`, provides the file path for the output. It should end with `.nii` or
  `.nii.gz`.

More details and other options can be found in the [command line tools](doc_command.md).
