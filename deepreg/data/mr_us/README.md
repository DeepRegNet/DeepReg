# MR UltraSound Data

Registration Requires following data:
- moving images
- moving labels
- fixed images
- fixed labels

As moving and fixed are of different modality, they are saved in different files.

Each moving/fixed image should have at least one label.

Two formats are supported: h5 files or nifti files.

## H5

Data are assumed to be stored together in four files under the same folder (default is `data/mr_us/h5/`) 
and the default file names are:
- `data/mr_us/h5/moving_images.h5`
- `data/mr_us/h5/moving_labels.h5`
- `data/mr_us/h5/fixed_images.h5`
- `data/mr_us/h5/fixed_labels.h5`

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

## Nifti
Training and test data are assumed to be stored separately under the same folder (default is `data/mr_us/nifti/`).
The default folders are
- `data/mr_us/nifti/train` for training data
- `data/mr_us/nifti/test` for test data

Under `train` or `test`, there are four folders saving images and labels:
- `moving_images.h5`
- `moving_labels`
- `fixed_images`
- `fixed_labels`

In each folder, samples are stored in the format of `*.nii.gz`, each files represents one sample.
The file names should be consistent across all folders.
For labels, the labels corresponding to one image sample is stored in one single file,
the shape could be `[dim1, dim2, dim3]` or `[dim1, dim2, dim3, dim4]` where `dim4` is the axis for labels.