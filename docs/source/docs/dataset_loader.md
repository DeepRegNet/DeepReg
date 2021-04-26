# Dataset Loader

## Dataset type

DeepReg provides six dataset loaders to support the following three different types of
datasets:

- **Paired images**

  Images are organized into moving and fixed image pairs.

  An example case is two-modalities intra-subject registration, such as registering one
  subject's MR image to the corresponding ultrasound image.

- **Unpaired images**

  Images may be considered independent samples.

  An example case is single-modality inter-subject registration, such as registering one
  CT image to another from different subjects.

- **Grouped images**

  Images are organized into multiple groups.

  An example case is single-modality intra-subject registration, such as registering
  time-series images within individual subjects, a group is one subject in this case.

For all three above cases, the images can be either unlabeled or labeled. A label is
represented by a boolean mask on the image, such as a segmentation of an anatomical
structure or landmark.

## Dataset requirements

To use the provided dataset loaders, other detailed images and labels requirements are
described in individual dataset loader sections. General requirements are described as
follows.

- Image

  - DeepReg currently supports 3D images. But images do not have to be of the same
    shape, and it will be resized to the required shape using linear interpolation.

  - Currently, DeepReg only supports images stored in Nifti files or H5 files. Check
    [Nifti_loader](https://github.com/DeepRegNet/DeepReg/blob/main/deepreg/dataset/loader/nifti_loader.py)
    and
    [h5_loader](https://github.com/DeepRegNet/DeepReg/blob/main/deepreg/dataset/loader/h5_loader.py)
    for more details.

  - **Images are automatically normalized** at per-image level: the intensity values x
    equals to `(x-min(x)+EPS) / (max(x)-min(x)+EPS)` so that its values are between
    [0,1]. Check `GeneratorDataLoader.data_generator` in
    [loader interface](https://github.com/DeepRegNet/DeepReg/blob/main/deepreg/dataset/loader/interface.py)
    for more details.

- Label

  - If an image is labeled, the label shape is recommended to be the same as the image
    shape. Otherwise, the resize might give unexpected behaviours. But each image can
    have more than one labels.<br>

    For instance, an image of shape `(dim1, dim2, dim3)`, its label shape can be
    `(dim1, dim2, dim3)` (single label) or `(dim1, dim2, dim3, num_labels)` (multiple
    labels).

  - **All labels are assumed to have values between [0, 1].** So DeepReg accepts binary
    segmentation masks or soft labels with float values between [0,1]. This is to
    prevent accidental use of non-one-hot encoding to represent multiple class labels.
    In case of multi labels, please use one-hot encoding to transform them into multiple
    channels such that each class has its own binary label.

  - When the images are paired, the moving and fixed images must have the same number of
    labels.

  - When there are multiple labels, it is assumed that the labels are ordered, such that
    the channel of index `label_idx` is the same anatomical or pathological structure.

  - Currently, if the data are labeled, each data sample must have at least one label.
    For missing labels or partially labelled data, consider using all-zero masks as a
    workaround.

  - See further discussion in [Label sampling](exp_label_sampling.html).

.. \_paired-images:

## Paired images

For paired images, each pair contains a moving image and a fixed image. Optionally,
corresponding moving label(s) and fixed label(s).

Specifically, given a pair of images

- When the image is unlabeled,
  - moving image of shape `(m_dim1, m_dim2, m_dim3)`
  - fixed image of shape `(f_dim1, f_dim2, f_dim3)`
- When the image is labeled and there is only one label,
  - moving image of shape `(m_dim1, m_dim2, m_dim3)`
  - fixed image of shape `(f_dim1, f_dim2, f_dim3)`
  - moving label of shape `(m_dim1, m_dim2, m_dim3)`
  - fixed label of shape `(f_dim1, f_dim2, f_dim3)`
- When the image is labeled and there are multiple labels,
  - moving image of shape `(m_dim1, m_dim2, m_dim3)`
  - fixed image of shape `(f_dim1, f_dim2, f_dim3)`
  - moving label of shape `(m_dim1, m_dim2, m_dim3, num_labels)`
  - fixed label of shape `(f_dim1, f_dim2, f_dim3, num_labels)`

### Sampling

For paired images, one epoch of the dataset iterates all the image pairs sequentially
with random orders. So each image pair is sampled once in each epoch with equal chance.
For validation or testing, the random seed is fixed to ensure consistency.

When an image has multiple labels, e.g. the segmentation of different organs in a CT
image, only one label will be sampled during training. In particular, only corresponding
labels will be sampled between a pair of moving and fixed images. In case of validation
or testing, instead of sampling one label per image, all labels will be iterated.

### Configuration

An example configuration for paired dataset is provided as follows.

```yaml
dataset:
  train:
    dir: "data/test/h5/paired/train" # folder containing data
    format: "h5" # nifti or h5
    labeled: true # true or false
  valid:
    dir: "data/test/h5/unpaired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/unpaired/test"
    format: "h5"
    labeled: true
  type: "paired" # value should be paired / unpaired / grouped
  moving_image_shape: [16, 16, 16] # value should be like [dim1, dim2, dim3]
  fixed_image_shape: [8, 8, 8] # value should be like [dim1, dim2, dim3]
```

where, the configuration can be split into common configurations that shared by all
dataset types and specific configurations for paired images:

- Common configurations
  - `dir/train` gives the directory containing training data. Same for `dir/valid` and
    `dir/test`.
  - `format` can only be Nifti or h5 currently.
  - `type` can be paired, unpaired or grouped, corresponding to the dataset type
    described above.
  - `labeled` is a boolean indicating if the data is labeled or not.
- Paired images configurations
  - `moving_image_shape` is the shape of moving images, a list of three integers.
  - `fixed_image_shape` is the shape of fixed images, a list of three integers.

Optionally, multiple dataset directories can be specified, such that the data will be
sampled from several directories, for instance:

```yaml
dataset:
  train:
    dir: # folders containing data
      - "data/test/h5/paired/train1"
      - "data/test/h5/paired/train2"
    format: "h5" # nifti or h5
    labeled: true # true or false
  valid:
    dir: "data/test/h5/unpaired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/unpaired/test"
    format: "h5"
    labeled: true
  type: "paired" # value should be paired / unpaired / grouped
  moving_image_shape: [16, 16, 16] # value should be like [dim1, dim2, dim3]
  fixed_image_shape: [8, 8, 8] # value should be like [dim1, dim2, dim3]
```

This is particularly useful when performing an
[experiment such as cross-validation](../tutorial/cross_val.html).

### File loader

For paired data, the specific requirements for data stored in Nifti and h5 files are
described as follows.

#### Nifti

Nifti data are stored in files with suffix `.nii.gz`. Each file should contain only one
3D or 4D tensor, corresponding to an image or a label.

`obs` is short for one observation of a data sample - a 3D image volume or a 3D/4D label
volume - and the name can be any string.

All image data should be placed under `moving_images/`, `fixed_images/` with respect to
the provided directory. The label data should be placed under `moving_labels/`, and
`fixed_labels/`, if available. These are _top_ directories.

File names should be consistent between top directories, e.g.:

- moving_images/
  - obs1.nii.gz
  - obs2.nii.gz
  - ...
- fixed_images/
  - obs1.nii.gz
  - obs2.nii.gz
  - ...
- moving_labels/
  - obs1.nii.gz
  - obs2.nii.gz
  - ...
- fixed_labels/
  - obs1.nii.gz
  - obs2.nii.gz
  - ...

Check
[test paired Nifti data](https://github.com/DeepRegNet/DeepReg/tree/main/data/test/nifti/paired)
as an example.

Optionally, the data may not be all saved directly under the top directory. They can be
further grouped in subdirectories as long as the data paths are consistent.

#### H5

H5 data are stored in files with suffix `.h5`. Hierarchical multi-level indexing is not
used. Each file should contain multiple key-value pairs and values are 3D or 4D tensors.
Each file is equivalent to a top folder in Nifti cases.

All image data should be stored in `moving_images.h5`, `fixed_images.h5`. The label data
should be stored in `moving_labels.h5`, and `fixed_labels.h5`, if available.

The keys should be consistent between files, e.g.:

- moving_images.h5 has keys:
  - "obs1"
  - "obs2"
  - ...
- fixed_images.h5 has keys:
  - "obs1"
  - "obs2"
  - ...
- moving_labels.h5 has keys:
  - "obs1"
  - "obs2"
  - ...
- fixed_labels.h5 has keys:
  - "obs1"
  - "obs2"
  - ...

Check
[test paired H5 data](https://github.com/DeepRegNet/DeepReg/tree/main/data/test/h5/paired)
as an example.

## Unpaired images

For unpaired images, all images are considered as independent and they must have the
same shape. Optionally, there are corresponding labels for the images.

Specifically,

- When the image is unlabeled,
  - image of shape `(dim1, dim2, dim3)`
- When the image is labeled and there is only one label,
  - image of shape `(dim1, dim2, dim3)`
  - label of shape `(dim1, dim2, dim3)`
- When the image is labeled and there are multiple labels,
  - image of shape `(dim1, dim2, dim3)`
  - label of shape `(dim1, dim2, dim3, num_labels)`

### Sampling

During each epoch, image pairs will be sampled without replacement. Therefore, given N
images, one epoch will thereby have floor(N / 2) image pairs. For validation or testing,
the random seed is fixed to ensure consistency.

In case of multiple labels, the sampling method is the same as in
[paired data](#sampling). In particular, the only corresponding label pairs will be
sampled between the two sampled images.

### Configuration

An example configuration for unpaired dataset is provided as follows.

```yaml
dataset:
  train:
    dir: "data/test/h5/paired/train" # folder containing data
    format: "h5" # nifti or h5
    labeled: true # true or false
  valid:
    dir: "data/test/h5/unpaired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/unpaired/test"
    format: "h5"
    labeled: true
  type: "unpaired" # value should be paired / unpaired / grouped
  image_shape: [16, 16, 16] # value should be like [dim1, dim2, dim3]
```

where

- Common configurations

  Same as [paired images](#configuration).

- Unpaired images configurations
  - `image_shape` is the shape of images, a list of three integers.

### File loader

For unpaired data, the specific requirements for data stored in nifti and h5 files are
described as follows.

#### Nifti

Nifti data are stored in files with suffix `.nii.gz` or `.nii`. Each file must contain
only one 3D or 4D tensor, corresponding to an image or a label.

`obs` is short for one observation of a data sample - a 3D image volume or a 3D/4D label
volume - and the name can be any string.

All image data should be placed under `images/`. The label data should be placed under
`labels/`, if available. These are _top_ directories.

File names should be consistent between top directories, e.g.:

- images/
  - obs1.nii.gz
  - obs2.nii.gz
  - ...
- labels/
  - obs1.nii.gz
  - obs2.nii.gz
  - ...

Check
[test unpaired Nifti data](https://github.com/DeepRegNet/DeepReg/tree/main/data/test/nifti/unpaired)
as an example.

#### H5

F5 data are stored in files with suffix `.h5`. Hierarchical multi-level indexing is not
used. Each file should contain multiple key-value pairs and values are 3D or 4D tensors.
Each file is equivalent to a top folder in Nifti cases.

All image data should be placed under `images.h5`. The label data should be placed under
`labels.h5`, if available.

The keys should be consistent between files, e.g.:

- images.h5 has keys:
  - "obs1"
  - "obs2"
  - ...
- labels.h5 has keys:
  - "obs1"
  - "obs2"
  - ...

Check
[test unpaired H5 data](https://github.com/DeepRegNet/DeepReg/tree/main/data/test/h5/unpaired)
as an example.

## Grouped images

For grouped images, images may not be paired but organized into multiple groups. Each
group must have at least two images.

The requirements are the same as unpaired images. Specifically,

- When the image is unlabeled,
  - image of shape `(dim1, dim2, dim3)`
- When the image is labeled and there is only one label,
  - image of shape `(dim1, dim2, dim3)`
  - label of shape `(dim1, dim2, dim3)`
- When the image is labeled and there are multiple labels,
  - image of shape `(dim1, dim2, dim3)`
  - label of shape `(dim1, dim2, dim3, num_labels)`

### Sampling

For sampling image pairs, DeepReg provides the following options:

- **inter-group sampling**, where the moving image and fixed image come from different
  groups.
- **intra-group sampling**, where the moving image and fixed image come from the same
  group.
- **mixed sampling**, where the image pairs are mixed from inter-group sampling and
  intra-group sampling.

For validation or testing, the random seed is fixed to ensure consistency.

In case of multiple labels, the sampling method is the same as [paired data](#sampling).
In particular, only the corresponding label pairs will be sampled between the two
sampled images.

#### Intra-group

To form image pairs, the group and image are sampled sequentially at two stages,

1. Sample a group from which the moving and fixed images will be sampled.
2. Sample two different images from the group as moving and fixed images.<br> When
   sampling images from the same group, there are multiple options, denoted by
   `intra_group_option`:
   - `forward`: the moving image always has a smaller image index than fixed image.
   - `backward`: the moving image always has a larger image index than fixed image.
   - `unconstrained`: no constraint on the image index as long as the two images are
     different.

Therefore, each epoch generates the same number of image pairs as the number of groups,
where all groups will be first shuffled and iterated. The `intra_group_option` is useful
in implementing temporal-order sensitive sampling strategy.

#### Inter-group

To form image pairs, the group and image are sampled sequentially at two stages,

1. Sample the first group, from which the moving image will be sampled.
2. Sample the second group, from which the fixed image will be sampled.
3. Sample an image from the first group as moving image.
4. Sample an image from the second group as fixed image.

Therefore, each epoch generates the same number of image pairs as the number of groups,
where all groups will be first shuffled and iterated.

#### Mixed

Optionally, it is possible to mix inter-group and intra-group sampling by specifying the
intra-group image sampling probability `intra_group_prob`=[0,1]. The value 0 means
entirely inter-group sampling and 1 means entirely intra-group sampling.

Given 0<p<1, when generating intra-group pairs, there is (1-p)\*100% chance to sample
the fixed images from a different group, after sampling the moving image from the
current intra-group images.

#### Iterated

Optionally, it is possible to generate all combinations of inter-/intra-group image
pairs, with `sample_image_in_group` set to false. This is originally designed for
evaluation. Mixing inter-/intra-group sampling is not supported with with
`sample_image_in_group` set to false.

### Configuration

An example configuration for grouped dataset is provided as follows.

```yaml
dataset:
  train:
    dir: "data/test/h5/paired/train" # folder containing data
    format: "h5" # nifti or h5
    labeled: true # true or false
  valid:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  type: "unpaired" # value should be paired / unpaired / grouped
  intra_group_prob: 1 # probability of intra-group sampling, value should be between 0 and 1
  intra_group_option: "forward" # option for intra-group sampling, value should be forward / backward / unconstrained
  sample_image_in_group: true # true if sampling one image per group, value should be true / false
  image_shape: [16, 16, 16] # value should be like [dim1, dim2, dim3]`
```

where

- Common configurations

  Same as [paired images](#configuration).

- Grouped images configurations
  - `intra_group_prob`, a value between 0 and 1, 0 is for inter-group only and 1 is for
    intra-group only.
  - `intra_group_option`, forward or backward or unconstrained, as described above.
  - `sample_image_in_group`, true if sampling one image at a time per group, false if
    generating all possible pairs.

### File loader

For grouped data, the specific requirements for data stored in Nifti and h5 files are
described as follows.

#### Nifti

Nifti data are stored in files with suffix `.nii.gz`. Each file should contain only one
3D or 4D tensor, corresponding to an image or a label.

`obs` is short for one observation of a data sample - a 3D image volume or a 3D/4D label
volume - and the name can be any string.

All image data should be placed under `images/`. The label data should be placed under
`labels/`, if available. These are _top_ directories.

The leaf directories will be considered as different groups, and file names should be
consistent between top directories, e.g.:

- images
  - group1
    - obs1.nii.gz
    - obs2.nii.gz
    - ...
  - ...
- labels
  - group1
    - obs1.nii.gz
    - obs2.nii.gz
    - ...
  - ...

Check
[test grouped Nifti data](https://github.com/DeepRegNet/DeepReg/tree/main/data/test/nifti/grouped)
as an example.

#### H5

H5 data are stored in files with suffix `.h5`. Hierarchical multi-level indexing is not
used. Each file should contain multiple key-value pairs and values are 3D or 4D tensors.
Each file is equivalent to a top folder in Nifti cases.

All image data should be placed under `images.h5`. The label data should be placed under
`labels.h5`, if available.

The keys must satisfy a specific format, `group-%d-%d`, where `%d` represents an integer
number. The first number corresponds to the group index, and the second number
corresponds to the observation index. For example, `group-3-2` corresponds to the second
observation from the third group.

The keys should be consistent between files, e.g.:

- images.h5 has keys:
  - "group-1-1"
  - "group-1-2"
  - ...
  - "group-2-1"
  - ...
- labels.h5 has keys:
  - "group-1-1"
  - "group-1-2"
  - ...
  - "group-2-1"
  - ...

Check
[test grouped H5 data](https://github.com/DeepRegNet/DeepReg/tree/main/data/test/h5/grouped)
as an example.
