# Nifti Data Format

The code implied the following assumptions

- Each `.nii.gz` file contains only one tensor, which is either image or label.
- Using `nib.load`, image's values are between 0 and 255 and label's value are between 0 and 1
- Image shape should be `[width, height, depth]`
- Label shape should be `[width, height, depth]` or `[width, height, depth, channels]` if multiple labels exist

The supported data set cases are:

- unpaired data with labels (case 1-1)
- unpaired data without labels (case 1-2)
- paired data with labels (case 2-1)
- paired data without labels (case 2-2)

The corresponding configuration should be
```yaml
data:
  dir:                   # required, directory of data under which we have train/images, etc.
  format: "nifti"
  paired:                # required, true if paired else false
  labeled:               # required, true if labeled else false
  moving_image_shape:    # required if paired, [dim1, dim2, dim3]
  fixed_image_shape:     # required if paired, [dim1, dim2, dim3]
  image_shape:           # required if unpaired or grouped, [dim1, dim2, dim3]
  intra_group_prob:      # required if grouped, value between [0,1], 0 means inter-group only and 1 means intra-group only
  intra_group_option:    # required if grouped, feasible values are: "forward", "backward", "unconstrained"
  sample_image_in_group: # required if grouped, false is want to generate all possible data pairs
                         # if false, intra_group_prob must be 0 or 1
```

and detailed requirements of the data organization are as follows.

## Case 1-1 Unpaired mages without labels

We only have images without any labels and all images are independent samples.
So all data should be stored under `train/images`, e.g.:

- train
  - images
    - subject1.nii.gz
    - subject2.nii.gz
    - ...

(It is also ok if the data are further grouped into different directories under `images`
as we will directly scan all nifti files under `train/images`.)

## Case 1-2 Unpaired images with labels

We have both images and labels.
So all images should be stored under `train/images` and all labels should be stored under `train/labels`.
The corresponding image file name and label file name should be exactly the same, e.g.:

- train
  - images
    - subject1.nii.gz
    - subject2.nii.gz
    - ...
  - labels
    - subject1.nii.gz
    - subject2.nii.gz
    - ...


## Case 2-1 Paired images without labels

We only have paired images without any labels.
So all data should be stored under `train/moving_images`, `train/fixed_images`
and the images corresponding to the same subject should have exactly the same name, e.g.:

- train
  - moving_images
    - subject1.nii.gz
    - subject2.nii.gz
    - ...
  - fixed_images
    - subject1.nii.gz
    - subject2.nii.gz
    - ...

(It is ok if the data are further grouped into different directories under `train/moving_images`
and `train/fixed_images` as we will directly scan all nifti files under them.)


## Case 2-2 Paired images with labels

We have both images and labels.
So all data should be stored under `train/moving_images`, `train/fixed_images`, `train/moving_labels`,
and `train/fixed_labels` .
The images and labels corresponding to the same subjects/groups should have exactly the same names, e.g.:

- train
  - moving_images
    - subject1.nii.gz
    - subject2.nii.gz
    - ...
  - fixed_images
    - subject1.nii.gz
    - subject2.nii.gz
    - ...
  - moving_labels
    - subject1.nii.gz
    - subject2.nii.gz
    - ...
  - fixed_labels
    - subject1.nii.gz
    - subject2.nii.gz
    - ...

## Case 3-1 Grouped images without labels

We have images without any labels, but images are grouped under different subjects/groups,
e.g. time-series observations for each subject/group.
For instance, the data set can be the CT scans of multiple patients (groups)
where each patient has multiple scans acquired at different time points.
So all data should be stored under `train/images` and the leaf directories
(directories that do not have sub-directories) must represent different groups, e.g.:

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

(It is also ok if the data are grouped into different directories,
but the leaf directories will be considered as different groups.)

## Case 3-2 Grouped images with labels

We have both images and labels.
So all images should be stored under `train/images` and all labels should be stored under `train/labels`.
The leaf directories will be considered as different groups
and the corresponding image file name and label file name should be exactly the same, e.g.:

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
