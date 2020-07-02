# Dataset Loader

## Existing Dataset Loaders

DeepReg provides multiple dataset loaders to support different scenarios.

### Dataset Structure

We support the following three types of dataset structure.

#### Image

##### 1. Paired Images

Images are organized into moving and fixed image pairs.

An example case is two-modalities intra-subject registration, such as registering one
subject's MR image to the corresponding ultrasound image.

##### 2. Unpaired Images

Images are independent samples.

An example case is single-modality inter-subject registration, such as registering one
CT image to another from different subjects.

##### 3. Grouped images

Images are organized into multiple groups.

An example case is single-modality intra-subject registration, such as registering
time-series images within individual subjects, a group is a subject in this case.

!> Currently, DeepReg only supports 3D images.

!> All images arerequired to have the same shape, `(m_dim1, m_dim2, m_dim3)`. <br> For
paired images, the moving images and fixed images can have different shapes,
`(m_dim1, m_dim2, m_dim3)` and `(f_dim1, f_dim2, f_dim3)`.

#### Label

For all three above cases, the data can be either unlabeled or labeled.

!> If an image is labeled, the label's shape has to be the same as the image. But each
image can have more than one labels. So given an image of shape `(dim1, dim2, dim3)`,
its label's shape can be `(dim1, dim2, dim3)` or `(dim1, dim2, dim3, num_labels)`.

!> Currently, if the data are labeled, each data sample has to have at least one label.
For missing labels, consider using all zeros as a work-around.

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

##### 3. Grouped images

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

##### 3. Grouped images

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
