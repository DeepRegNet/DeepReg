

# Data folder structure for nifti images 

In the following, we take train directory as an example to list how the files should be stored.

## Nifti Data Format

Assuming each `.nii.gz` file contains only one tensor, which is either image or label.

### Unpaired data
This is the simplest case. Data are assumed to be stored under `train/images` and `train/labels` directories.

#### Nifti Case 1-1 Images only

We only have images without any labels and all images are considered to be independent samples. So all data should be stored under `train/images`, e.g.:

- train
  - images
    - obs1.nii.gz
    - obs2.nii.gz
    - ...

(It is also ok if the data are further grouped into different directories under `images` as we will directly scan all nifti files under `train/images`.)

#### Nifti Case 1-2 Images with labels

In this case, we have both images and labels. So all images should be stored under `train/images` and all labels should be stored under `train/labels`. _The corresponding image file name and label file name should be exactly the same_, e.g.:

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

We have images without any labels, but images are grouped under different groups, e.g. time-series observations for each subject (group. For instance, the data set can be the CT scans of multiple patients (groups) where each patient has multiple scans acquired at different time points. So all data should be stored under `train/images` and the leaf directories (directories that do not have sub-directories) must represent different groups, e.g.:

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

(It is also ok if the data are grouped into different directories, but the leaf directories will be considered as different groups.)


#### Nifti Case 2-2 Images with labels

We have both images and labels. So all images should be stored under `train/images` and all labels should be stored under `train/labels`.  The leaf directories will be considered as different groups and the corresponding image file name and label file name should be exactly the same, e.g.:

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

In this case, images are paired, for example, to represent a multimodal moving and fixed image pairs to register. Data are assumed to be stored under `train/moving_images`, `train/fixed_images`, `train/moving_labels`, and `train/fixed_labels` directories.

#### Nifti Case 3-1 Images only

We only have paired images without any labels. So all data should be stored under `train/moving_images`, `train/fixed_images` and the images corresponding to the same group should have exactly the same name, e.g.:

- train
  - moving_images
    - obs1.nii.gz
    - obs2.nii.gz
    - ...
  - fixed_images
    - obs1.nii.gz
    - obs2.nii.gz
    - ...

(It is ok if the data are further grouped into different directories under `train/moving_images` and `train/fixed_images` as we will directly scan all nifti files under them.)


#### Nifti Case 3-2 Images with labels

We have both images and labels. So all data should be stored under `train/moving_images`, `train/fixed_images`, `train/moving_labels`, and `train/fixed_labels` . The images and labels corresponding to the same groups should have exactly the same names, e.g.:

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

