# Nifti Data Format

The code implied the following assumptions
- Each `.nii.gz` file contains only one tensor, which is either image or label.
- Using `nib.load`, image's values are between 0 and 255 and label's value are between 0 and 1
- Image shape should be `[width, height, depth]`
- Label shape should be `[width, height, depth]` or `[width, height, depth, channels]` if multiple labels exist
