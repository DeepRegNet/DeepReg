# Dataset Loader

## Existing Dataset Loaders

DeepReg provides multiple dataset loaders to support different scenarios.

### Structure

We support the following three types of structure.

#### Image

##### 1. Paired Images

Images are organized into moving and fixed image pairs.

An example case is two-modalities intra-subject registration, such as registering one
subject's MR image to the corresponding ultrasound image.

##### 2. Unpaired Images

Images are independent samples.

An example case is single-modality inter-subject registration, such as registering one
CT image to another from different subjects.

##### 3. Grouped Images

Images are organized into multiple groups.

An example case is single-modality intra-subject registration, such as registering
time-series images within individual subjects, a group is a subject in this case.

!> Currently, DeepReg only supports 3D images.<br> So all images arerequired to have the
same shape, `(m_dim1, m_dim2, m_dim3)`. <br> For 　 paired images, the moving images and
fixed images can have different shapes, `(m_dim1, m_dim2, m_dim3)` and
`(f_dim1, f_dim2, f_dim3)`.

#### Label

For all three above cases, the data can be either unlabeled or labeled.

However, there are multiple requirements for the labels.

- If an image is labeled, the label's shape has to be the same as the image. But each
  image can have more than one labels. So given an image of shape `(dim1, dim2, dim3)`,
  its label's shape can be `(dim1, dim2, dim3)` or `(dim1, dim2, dim3, num_labels)`.
- When the images are paired, the moving and fixed images must have the same number of
  labels.
- When there are multiple labels, it is the user's responsibility to ensure the labels
  are ordered, such that the channel of index `label_idx` is the same anatomical or
  pathological structure. This is important to assure the consistency between the moving
  and fixed labels specially when the images are not paired.

!> Currently, if the data are labeled, each data sample has to have at least one label.
For missing labels, consider using all zeros as a work-around.

### Sampling

As the registration network takes only one pair of images (and the corresponding labels)
at a time, some sampling strategy has to be used in case of not paired data or multiple
labels.

#### Image Sampling

We describe the sampling method during training as follows. The sampling results are
different for each epoch. For validation or testing, the random seed will be fixed to
ensure consistency.

##### 1. Paired Images

As images have been paired, one epoch of the dataset iterates all the pairs sequentially
with random orders. So each image pair is sampled once in each epoch with equal chance.

##### 2. Unpaired Images

To form pairs, all images will be first shuffled and then the pairs are formed
two-by-two. Assuming there are N images, one epoch will thereby have floor(N / 2) image
pairs. Equivalently, this can be considered as a sampling without replacement.

##### 3. Grouped Images

For grouped images, there are two basic sampling methods:

- **inter-group sampling**, where the moving image and fixed image come from different
  groups.<br> For each epoch, the groups are
- **intra-group sampling**, where the moving image and fixed image come from the same
  group.

###### Inter-group

To form pairs, we sample the group and image sequentially,

1. Sample a group i as the group of moving image.
2. Sample a group j != i as the group of fixed image.
3. Sample an image from the group i as moving image.
4. Sample an image from the group j as fixed image.

Assuming there are G groups, each epoch generates G pairs where all groups will be first
shuffled and iterated as the group of moving image.

###### Intra-group

To form pairs, we sample the group and image sequentially,

1. Sample a group as the group of moving image.
2. Sample two different images from the group as moving and fixed images.<br> When
   sampling images from the same group, there are multiple options, denoted by
   `intra_group_option`:
   - `forward`: the moving image always has a smaller image index than fixed image.
   - `backward`: the moving image always has a larger image index than fixed image.
   - `unconstrained`: no constraint on the image index as long as the two images are
     different.

Assuming there are G groups, each epoch generates G pairs where all groups will be first
shuffled and iterated as the group of moving image.

###### Mixing inter-group and intra-group

Optionally, it is possible to mix inter-group and intra-group sampling by specifying the
intra-groupe image sampling probability `intra_group_prob`=[0,1]. The value 0 means
entirely inter-group sampling and 1 means entirely intra-group sampling.

Given 0<p<1, when generating intra-group pairs, there is (1-p)\*100% chance to sample
the fixed images from a different group, after sampling the moving image from the
current intra-group images.

###### Iterating all image pairs

Optionally, it is possible to generate all combination of inter-/intra-group image
pairs, with `sample_image_in_group` set to false. This is originally designed for
evaluation and mixing inter-/intra-group sampling is not supported.

#### Label Sampling

When an image has multiple labels, e.g. the segmentations of different organs in a CT
image, only one label will be sampled during training. However, all labels will be
sampled for validation or testing.

!> This is currently fixed by default and impossible to configure.

### Dataset Configuration

To use the existing dataset loaders, specific configurations should be given
corresponding to different dataset structures.

#### Common Configuration

First they all share a common configuration

```yaml
dataset:
  dir:
    train: "data/test/h5/paired/train" # folder saving training data
    valid: "data/test/h5/paired/test" # folder saving validation data
    test: "data/test/h5/paired/test" # folder saving test data
  format: "nifti" # value should be nifti / h5
  type: "paired" # value should be paired / unpaired / grouped
  labeled: true # value should be true / false
```

where

- The `dir/train` gives the directory containing training data. Same for `dir/valid` and
  `dir/test`.
- The `format` can be only nifti or h5 currently.
- The `type` can be paired, unpaired or grouped, corresponding to the dataset structures
  described above.
- The `labeled` is a boolean indicating if the data is labeled or not.

Optionally, for the directories of data, we can provide more than one directories, for
instance:

```yaml
dataset:
  dir:
    train: # folder saving training data
      - "data/test/h5/paired/train1"
      - "data/test/h5/paired/train2"
    valid: "data/test/h5/paired/test" # folder saving validation data
    test: "data/test/h5/paired/test" # folder saving test data
```

with this configuration, the data will come from two directories.

#### Paired Data

For paired data, apart from the common configuration, `moving_image_shape` and
`fixed_image_shape` have to be defined. For example:

```yaml
dataset:
  dir:
    train: "data/test/h5/paired/train" # folder saving training data
    valid: "data/test/h5/paired/test" # folder saving validation data
    test: "data/test/h5/paired/test" # folder saving test data
  format: "nifti" # value should be nifti / h5
  type: "paired" # value should be paired / unpaired / grouped
  labeled: true # value should be true / false
  moving_image_shape: [64, 64, 60] # value should be like [dim1, dim2, dim3]
  fixed_image_shape: [44, 59, 41] # value should be like [dim1, dim2, dim3]
```

#### Unpaired Data

For unpaired data, apart from the common configuration, `image_shape` has to be defined.
For example:

```yaml
dataset:
  dir:
    train: "data/test/h5/paired/train" # folder saving training data
    valid: "data/test/h5/paired/test" # folder saving validation data
    test: "data/test/h5/paired/test" # folder saving test data
  format: "nifti" # value should be nifti / h5
  type: "unpaired" # value should be paired / unpaired / grouped
  labeled: true # value should be true / false
  image_shape: [64, 64, 60] # value should be like [dim1, dim2, dim3]
```

#### Grouped Data

For grouped data, apart from the common configuration, as unpaired data `image_shape`
has to be defined.

There are three other parameters:

- `intra_group_prob`, a value between 0 and 1, 0 is for inter-group only and 1 is for
  intra-group only.
- `intra_group_option`,　 forward or backward or unconstrained, as described above.
- `sample_image_in_group`, true if sampling one image at a time per group, false if
  generate all pairs.

For example:

```yaml
dataset:
  dir:
    train: "data/test/h5/paired/train" # folder saving training data
    valid: "data/test/h5/paired/test" # folder saving validation data
    test: "data/test/h5/paired/test" # folder saving test data
  format: "nifti" # value should be nifti / h5
  type: "unpaired" # value should be paired / unpaired / grouped
  labeled: true # value should be true / false
  intra_group_prob: 1 # probability of intra-group sampling, value should be between 0 and 1
  intra_group_option: "forward" # option for intra-group sampling, value should be forward / backward / unconstrained
  sample_image_in_group: true # true if sampling one image per group, value should be true / false
  image_shape: [64, 64, 60] # value should be like [dim1, dim2, dim3]
```

### Data Format

Deepreg supports currently Nifti and h5 file format. Each dataset structure has specific
requirements for the data storage.

#### Nifti Data

Nifti data are stored in files with suffix `.nii.gz`. Each file should contain only one
3D or 4D tensor, corresponding to an image or a label.

The requirements for different structures are as follows. `obs` is short for one
observation of a data sample - a 3D image volume or a 3D/4D label volume - and the name
can be any string.

##### 1. Paired Images

All image data should be placed under `moving_images/`, `fixed_images/`. The label data
should be placed under `moving_labels/`, and `fixed_labels/`, if available. These are
_top_ directories.

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
[test paired data](https://github.com/ucl-candi/DeepReg/tree/master/data/test/nifti/paired)
as an example.

?> The data do not have to be all saved directly under the top directory. They can be
grouped in subdirectories as long as the data paths are consistent.

##### 2. Unpaired Images

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
[test unpaired data](https://github.com/ucl-candi/DeepReg/tree/master/data/test/nifti/unpaired)
as an example.

##### 3. Grouped Images

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
[test unpaired data](https://github.com/ucl-candi/DeepReg/tree/master/data/test/nifti/grouped)
as an example.

#### H5 Data

Nifti data are stored in files with suffix `.h5`. Hierarchical multi-level indexing is
not used. Each file should contain multiple key-value pairs and values are 3D or 4D
tensors. Each file is equivalent to a top folder in Nifti cases.

The requirements for different structures are as follows.

##### 1. Paired Images

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
[test paired data](https://github.com/ucl-candi/DeepReg/tree/master/data/test/h5/paired)
as an example.

##### 2. Unpaired Images

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
[test unpaired data](https://github.com/ucl-candi/DeepReg/tree/master/data/test/h5/unpaired)
as an example.

##### 3. Grouped Images

All image data should be placed under `images.h5`. The label data should be placed under
`labels.h5`, if available.

The keys have to satisfy a specific format, `group-%d-%d`, where `%d` represents an
integer number. The first number corresponds to the group index, and the second number
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
[test unpaired data](https://github.com/ucl-candi/DeepReg/tree/master/data/test/h5/grouped)
as an example.

## Customized Data Loader

(under development)
