# Deep Registration


## Environment Setup

The easiest way to install the python environment is with Miniconda/Anaconda.
- Use `conda create --name py36-tf2 python=3.6 tensorflow-gpu` to setup the latest gpu-support with tf.
- Use `pip install -r requirements.txt` to install the rest requirements

## Train

The training script is `train.py` and it accepts the following parameters
- `-g` or `--gpu`, **required**, providing available GPU indices, e.g. `-g 0` uses GPU of index 0 and `-g "0,1"` uses GPU of index 0 and 1.
- `-c` or `--config`, providing the path of the configuration file, `-c demo.yaml`. If not provided, the default configuration file `src/config/default.yaml` will be used.
- `-m` or `--memory`, providing this flag will prevent tensorflow to reserve all available GPU memory.
- `--ckpt`, providing the checkpoint to load, to prevent start training from random initialization. The path must ended in `.ckpt`, e.g. `--ckpt logs/demo/save/weights-epoch100.ckpt`
- `-l` or `--log`, providing the name of log folder. It not provided, a timestamp based folder name will be used.

For example, `python train.py -c demo.yaml -l demo -g "0,1"` will launch a training using the configuration file `demo.yaml` with GPU of index 0 and 1 and the log will be saved under `logs/demo`.

## Predict

The prediction script is `predict.py` and it accepts the following parameters
- `-g` or `--gpu`, **required**, providing available GPU indices, e.g. `-g 0` uses GPU of index 0. Multi-GPU setting is not tested for prediction.
- `--ckpt`, **required**, providing the checkpoint to load, to prevent start training from random initialization. The path must ended in `.ckpt`, e.g. `--ckpt logs/demo/save/weights-epoch100.ckpt`. When loading the checkpoint, a backup configuration file wil be used, in the example above, the configuration file will be under `logs/demo/`. 
- `-m` or `--memory`, providing this flag will prevent tensorflow to reserve all available GPU memory.
- `--bs`, providing the batch size, the number of data samples must be divided evenly by batch size during prediction. If not provided, batch size of 1 will be used.
- `--train`, providing this flag will execute prediction on the training data.

For example, `python predict.py -g 0 --ckpt logs/demo/save/weights-epoch100.ckpt` will launch a prediction using the provided checkpoint with one GPU. The results will be saved under `logs` in a new folder, in which a `metric.log` will be generated. An example of `metric.log` will be like

```
image 0, label 0, dice 0.7821, dist 6.9741
image 0, label 1, dice 0.0147, dist 7.1142
image 0, label 2, dice 0.0000, dist 13.4373
image 0, label 3, dice 0.0000, dist 14.0168
image 0, label 4, dice 0.0000, dist 14.0412
image 0, label 5, dice 0.0000, dist 18.5782
image 1, label 0, dice 0.8591, dist 1.4560
image 1, label 1, dice 0.0000, dist 9.1976
image 1, label 2, dice 0.0000, dist 8.4542
image 1, label 3, dice 0.0000, dist 16.8208
image 1, label 4, dice 0.0000, dist 5.3310
image 1, label 5, dice 0.0000, dist 27.9396

```

## Data

Required data contain four parts:
- moving images
- moving labels
- fixed images
- fixed labels

Each moving/fixed image should have at least one label.

Two formats are supported: h5 files or nifti files.

### H5

Data are assumed to be stored together in four files under the same folder (default is `data/h5/`) 
and the default file names are:
- `data/h5/moving_images.h5`
- `data/h5/moving_labels.h5`
- `data/h5/fixed_images.h5`
- `data/h5/fixed_labels.h5`

For images file, there is no constraint on the keys of h5 file. Each key corresponds to one image sample, 
the shape is assumed to be the same 3D shape among all samples `[dim1, dim2, dim3]`.

For labels file, the key is assumed to contain the corresponding image key in a specific format.
> For example, if `moving_images.h5` contains a key `case000025` and this image has multiple labels, 
> then the keys for labels should be `case000025_bin000`,`case000025_bin001`, etc.
> The values after `_bin` are not important as they will be sorted.
> But the label key must contain image key followed by `_bin`. 
> Moreover, the first label is assumed to be of the same type (one metric in tensorboard depends on this).

Each key corresponds to one label for  one image, and the shape should be the same as the image, 
i.e. `[dim1, dim2, dim3]`.

### Nifti
Training and test data are assumed to be stored separately under the same folder (default is `data/nifti/`).
The default folders are
- `data/nifti/train` for training data
- `data/nifti/test` for test data

Under `train` or `test`, there are four folders saving images and labels:
- `moving_images.h5`
- `moving_labels`
- `fixed_images`
- `fixed_labels`

In each folder, samples are stored in the format of `*.nii.gz`, each files represents one sample.
The file names should be consistent across all folders.
For labels, the labels corresponding to one image sample is stored in one single file,
the shape could be `[dim1, dim2, dim3]` or `[dim1, dim2, dim3, dim4]` where `dim4` is the axis for labels.