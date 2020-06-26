# Dataset configurations

A typical yaml config file is explained as follows.

```yaml
dataset:
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

