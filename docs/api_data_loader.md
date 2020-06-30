# How to arrange data files and folders to use predefined data loaders

Cutomised data loaders can be added to work with the core `DeepReg` algorithms, examples
are explained in [How to add a new data loader](/add_loader.md).

Currently, six use _scenarios_ are supported for unpaired, grouped and paired images,
each with two data loaders depends on whether corresponding labels are avaialble.
Details of sampling avaialble in these loaders are explained in
[Data sampling options](tutorial_sampling.md).

## Supported scenarios

### Unpaired images (e.g. single-modality inter-subject registration)

- Case 1-1 multiple independent images.
- Case 1-2 multiple independent images and corresponding labels.

### Grouped unpaired images (e.g. single-modality intra-subject registration)

- Case 2-1 multiple subjects each with multiple images.
- Case 2-2 multiple subjects each with multiple images and corresponding labels.

### Paired images (e.g. two-modality intra-subject registration)

- Case 3-1 multiple paired images.
- Case 3-2 multiple paired images and corresponding labels.

## Data Format

There are some prerequisites on the data:

- Data must be split into train / val / test before and placed in different directories.
  Although val or test data are optional.
- Each image or label is in 3D. Image has shape `(width, height, depth)`; label has
  shape `(width, height, depth)` or `(width, height, depth, num_labels)`.
- The data do not have to be of the same shape - All will be resized to the same shape
  before feed-in. In order to prevent unexpected effects, it may be recommended that all
  images are pre-processed to the desirable shape.

These predefined data loaders require specific _data folder structures_, which are
explained as follows, for nifti and h5 file formats, respectively.

### Data folder structure for nifti images

[Requirement](tutorial/predefined_loader_nifti.md)
[Example nifiti data folder](../data/test/nifti)

### Data folder structure for h5 images

[Requirement](tutorial/predefined_loader_h5.md)
[Example h5 data folder](../data/test/h5)

## How to use these predefined loaders in an experiment.

## Training and validation use the same data folder structure. Prediction can use a different test data folder structure.
