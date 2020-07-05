# Dataset Loader

## Dataset Type

DeepReg provides multiple dataset loaders to support the following three different types
of datasets:

- **Paired images**

  Images are organized into moving and fixed image pairs.

  An example case is two-modalities intra-subject registration, such as registering one
  subject's MR image to the corresponding ultrasound image.

  Read [paired images](#paired-images) for more details.

- **Unpaired images**

  Images are independent samples.

  An example case is single-modality inter-subject registration, such as registering one
  CT image to another from different subjects.

  Read [unpaired images](#unpaired-images) for more details.

- **Grouped images**

  Images are organized into multiple groups.

  An example case is single-modality intra-subject registration, such as registering
  time-series images within individual subjects, a group is a subject in this case.

  Read [grouped images](#grouped-images) for more details.

For all three above cases, the images can be either unlabeled or labeled. An label is
considered as a boolean mask on the image, it can be a segmentation or a one-hot
landmark.

## Dataset Requirements

To use the provided dataset loaders, there are multiple requirements for the images and
labels. Read each dataset loader for more details.

- Image

  - Currently, DeepReg only supports 3D images and all images are required to have the
    same shape, e.g. `(m_dim1, m_dim2, m_dim3)`.<br> Except For paired images, the
    moving images and fixed images can have different shapes, e.g.
    `(m_dim1, m_dim2, m_dim3)` and `(f_dim1, f_dim2, f_dim3)`.

- Label

  - If an image is labeled, the label's shape has to be the same as the image. But each
    image can have more than one labels.<br>

    For instance, an image if of shape `(dim1, dim2, dim3)`, its label's shape can be
    `(dim1, dim2, dim3)` (single label) or `(dim1, dim2, dim3, num_labels)` (multiple
    labels).

  - When the images are paired, the moving and fixed images must have the same number of
    labels.

  - When there are multiple labels, it is assumed that the labels are ordered, such that
    the channel of index `label_idx` is the same anatomical or pathological structure.

  - Currently, if the data are labeled, each data sample has to have at least one label.
    For missing labels, consider using all zeros as a work-around.

## Paired images

For paired images, each pair contains moving image and fixed image. Optionally, there
are corresponding moving label and fixed label.

Precisely, given a pair of images

- When the image is unlabeled, we have
  - moving image of shape `(m_dim1, m_dim2, m_dim3)`
  - fixed image of shape `(f_dim1, f_dim2, f_dim3)`
- When the image is labeled and there is only one label, we have
  - moving image of shape `(m_dim1, m_dim2, m_dim3)`
  - fixed image of shape `(f_dim1, f_dim2, f_dim3)`
  - moving label of shape `(m_dim1, m_dim2, m_dim3)`
  - fixed label of shape `(f_dim1, f_dim2, f_dim3)`
- When the image is labeled and there are multiple labels, we have
  - moving image of shape `(m_dim1, m_dim2, m_dim3)`
  - fixed image of shape `(f_dim1, f_dim2, f_dim3)`
  - moving label of shape `(m_dim1, m_dim2, m_dim3, num_labels)`
  - fixed label of shape `(f_dim1, f_dim2, f_dim3, num_labels)`

### Sampling

As the registration network takes only one pair of images (and the corresponding labels)
at a time, some sampling strategy has to be used in case of multiple labels.

During training, the sampling results are different for each epoch. For validation or
testing, the random seed is fixed to ensure consistency.

For paired images, one epoch of the dataset iterates all the pairs sequentially with
random orders. So each image pair is sampled once in each epoch with equal chance.

When an image has multiple labels, e.g. the segmentations of different organs in a CT
image, only one label will be sampled during training. In particular, the sampled label
will be always the same for moving and fixed images. In case of validation or testing,
instead of sampling one label per image, all labels will be iterated.

### Configuration

Therefore, we use the following configuration for paired dataset.

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

where, the configuration can be split into common configurations that shared by all
dataset types and specific configurations for paired images:

- Common configurations
  - The `dir/train` gives the directory containing training data. Same for `dir/valid`
    and `dir/test`.
  - The `format` can be only nifti or h5 currently. More details
  - The `type` can be paired, unpaired or grouped, corresponding to the dataset type
    described above.
  - The `labeled` is a boolean indicating if the data is labeled or not.
- Paired images configurations
  - The `moving_image_shape` is the shape of moving images, a list of three integers.
  - The `fixed_image_shape` is the shape of fixed images, a list of three integers.

Optionally, we can provide more than two or more directories, so that the data come from
multiple directories, for instance:

```yaml
dataset:
  dir:
    train: # folder saving training data
      - "data/test/h5/paired/train1"
      - "data/test/h5/paired/train2"
    valid: "data/test/h5/paired/test" # folder saving validation data
    test: "data/test/h5/paired/test" # folder saving test data
```

### File loader

For paired data, the specific requirements for data stored in nifti and h5 files are
described as follows.

#### Nifti

Nifti data are stored in files with suffix `.nii.gz`. Each file should contain only one
3D or 4D tensor, corresponding to an image or a label.

The requirements for different structures are as follows. `obs` is short for one
observation of a data sample - a 3D image volume or a 3D/4D label volume - and the name
can be any string.

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
[test paired data](https://github.com/ucl-candi/DeepReg/tree/master/data/test/nifti/paired)
as an example.

Optionally, the data do not have to be all saved directly under the top directory. They
can be grouped in subdirectories as long as the data paths are consistent.

#### H5

F5 data are stored in files with suffix `.h5`. Hierarchical multi-level indexing is not
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
[test paired data](https://github.com/ucl-candi/DeepReg/tree/master/data/test/h5/paired)
as an example.

## Unpaired images

For unpaired images, all images are considered as independent and they should have the
same shape. Optionally, there are corresponding labels for the images.

Precisely,

- When the image is unlabeled, we have
  - image of shape `(dim1, dim2, dim3)`
- When the image is labeled and there is only one label, we have
  - image of shape `(dim1, dim2, dim3)`
  - label of shape `(dim1, dim2, dim3)`
- When the image is labeled and there are multiple labels, we have
  - image of shape `(dim1, dim2, dim3)`
  - label of shape `(dim1, dim2, dim3, num_labels)`

### Sampling

As the registration network takes only one pair of images (and the corresponding labels)
at a time, some sampling strategy has to be used to pair the images and also for
multiple labels.

During training, the sampling results are different for each epoch. For validation or
testing, the random seed is fixed to ensure consistency.

To form pairs, all images will be first shuffled and then the pairs are formed
two-by-two. Assuming there are N images, one epoch will thereby have floor(N / 2) image
pairs. Equivalently, this can be considered as a sampling without replacement.

In case of multiple labels, the sampling method is the same as [paired data](#sampling).
In particular, the sampled label will be always the same for the two chosen images.

### Configuration

We use the following configuration for unpaired dataset.

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

where

- Common configurations

  Same as [paired images](#configuration).

- Unpaired images configurations
  - The `image_shape` is the shape of images, a list of three integers.

### File loader

For unpaired data, the specific requirements for data stored in nifti and h5 files are
described as follows.

#### Nifti

Nifti data are stored in files with suffix `.nii.gz`. Each file should contain only one
3D or 4D tensor, corresponding to an image or a label.

The requirements for different structures are as follows. `obs` is short for one
observation of a data sample - a 3D image volume or a 3D/4D label volume - and the name
can be any string.

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
[test unpaired data](https://github.com/ucl-candi/DeepReg/tree/master/data/test/h5/unpaired)
as an example.

## Grouped images

For grouped images, all images are unpaired but organized into multiple groups. Each
group has to have at least two images.

The requirements are the same as unpaired images. Precisely,

- When the image is unlabeled, we have
  - image of shape `(dim1, dim2, dim3)`
- When the image is labeled and there is only one label, we have
  - image of shape `(dim1, dim2, dim3)`
  - label of shape `(dim1, dim2, dim3)`
- When the image is labeled and there are multiple labels, we have
  - image of shape `(dim1, dim2, dim3)`
  - label of shape `(dim1, dim2, dim3, num_labels)`

### Sampling

As the registration network takes only one pair of images (and the corresponding labels)
at a time, some sampling strategy has to be used to pair the images and also for
multiple labels.

During training, the sampling results are different for each epoch. For validation or
testing, the random seed is fixed to ensure consistency.

In case of multiple labels, the sampling method is the same as [paired data](#sampling).
In particular, the sampled label will be always the same for the two chosen images.

For images, DeepReg mainly provides the following sampling methods:

- **inter-group sampling**, where the moving image and fixed image come from different
  groups.<br> For each epoch, the groups are
- **intra-group sampling**, where the moving image and fixed image come from the same
  group.
- **mixed sampling**, where the image pairs are mixed from inter-group sampling and
  intra-group sampling.

#### Inter-group

To form pairs, we sample the group and image sequentially,

1. Sample a group, denoted by A, as the group of moving image.
2. Sample another different group, denoted by B, as the group of fixed image.
3. Sample an image from the group A as moving image.
4. Sample an image from the group B as fixed image.

Assuming there are G groups, each epoch generates G pairs where all groups will be first
shuffled and iterated as the group of moving image.

#### Intra-group

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

#### Mixed

Optionally, it is possible to mix inter-group and intra-group sampling by specifying the
intra-groupe image sampling probability `intra_group_prob`=[0,1]. The value 0 means
entirely inter-group sampling and 1 means entirely intra-group sampling.

Given 0<p<1, when generating intra-group pairs, there is (1-p)\*100% chance to sample
the fixed images from a different group, after sampling the moving image from the
current intra-group images.

#### Iterated

Optionally, it is possible to generate all combination of inter-/intra-group image
pairs, with `sample_image_in_group` set to false. This is originally designed for
evaluation and mixing inter-/intra-group sampling is not supported.

### Configuration

We use the following configuration for grouped dataset.

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

where

- Common configurations

  Same as [paired images](#configuration).

- Grouped images configurations
  - `intra_group_prob`, a value between 0 and 1, 0 is for inter-group only and 1 is for
    intra-group only.
  - `intra_group_option`,　 forward or backward or unconstrained, as described above.
  - `sample_image_in_group`, true if sampling one image at a time per group, false if
    generate all pairs.

### File loader

For grouped data, the specific requirements for data stored in nifti and h5 files are
described as follows.

#### Nifti

Nifti data are stored in files with suffix `.nii.gz`. Each file should contain only one
3D or 4D tensor, corresponding to an image or a label.

The requirements for different structures are as follows. `obs` is short for one
observation of a data sample - a 3D image volume or a 3D/4D label volume - and the name
can be any string.

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

#### H5

F5 data are stored in files with suffix `.h5`. Hierarchical multi-level indexing is not
used. Each file should contain multiple key-value pairs and values are 3D or 4D tensors.
Each file is equivalent to a top folder in Nifti cases.

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
