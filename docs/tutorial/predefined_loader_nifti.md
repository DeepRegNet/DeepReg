# Data folder structure for nifti images

In the following, train directory is used as an example to list how the files should be
placed.

## Nifti Data Format

Assuming each `.nii.gz` file contains only one tensor, which is either an image or a
label.

### Unpaired data

This is the simplest case. Data are assumed to be placed under `train/images` and
`train/labels` directories.

#### Nifti Case 1-1 Images only

All images should be placed under `train/images`, e.g.:

- train
  - images
    - obs1.nii.gz
    - obs2.nii.gz
    - ...

_The data can be further grouped into different directories under `images` nifti files
under `train/images` will be scanned and included._

#### Nifti Case 1-2 Images with labels

In this case, all images should be placed under `train/images` and all labels should be
placed under `train/labels`. _The corresponding image file name and label file name
should be exactly the same_, e.g.:

- train
  - images
    - obs1.nii.gz
    - obs2.nii.gz
    - ...
  - labels
    - obs1.nii.gz
    - obs2.nii.gz
    - ...

### Grouped unpaired images

#### Nifti Case 2-1 Images only

Images are grouped under different groups, e.g. time-series observations for each
subject For instance, the data set can be the CT scans of multiple patients (groups)
where each patient has multiple scans acquired at different time points. All data should
be placed under `train/images`. _The leaf directories (directories that do not have
sub-directories) must represent different groups_, e.g.:

- train
  - images
    - group1
      - obs1.nii.gz
      - obs2.nii.gz
      - ...
    - group2
      - obs1.nii.gz
      - obs2.nii.gz
      - ...
    - ...

#### Nifti Case 2-2 Images with labels

All images should be placed under `train/images` and all labels should be placed under
`train/labels`. _The leaf directories will be considered as different groups and the
corresponding image file name and label file name should be exactly the same_, e.g.:

- train
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

### Paired images

In this case, images are paired, for example, to represent a multimodal moving and fixed
image pairs to register. Data are placed under `train/moving_images`,
`train/fixed_images`, `train/moving_labels`, and `train/fixed_labels` directories.

#### Nifti Case 3-1 Images only

All image data should be placed under `train/moving_images`, `train/fixed_images` and
the images corresponding to the same group should have exactly the same name, e.g.:

- train
  - moving_images
    - obs1.nii.gz
    - obs2.nii.gz
    - ...
  - fixed_images
    - obs1.nii.gz
    - obs2.nii.gz
    - ...

_The data can be further grouped into different directories under `train/moving_images`
and `train/fixed_images` as we will directly scan all nifti files under them._

#### Nifti Case 3-2 Images with labels

All image and label data should be placed under `train/moving_images`,
`train/fixed_images`, `train/moving_labels`, and `train/fixed_labels`. _The images and
labels corresponding to the same groups should have exactly the same names_, e.g.:

- train
  - moving_images
    - obs1.nii.gz
    - obs2.nii.gz
    - ...
  - fixed_images
    - obs1.nii.gz
    - obs2.nii.gz
    - ...
  - moving_labels
    - obs1.nii.gz
    - obs2.nii.gz
    - ...
  - fixed_labels
    - obs1.nii.gz
    - obs2.nii.gz
    - ...
