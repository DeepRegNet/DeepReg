# Dataset configuration

A typical yaml config file is explained as follows.

```yaml
dataset:
  dir: # required, data directory, e.g. "data/test/nifti/grouped"
  mode: "train" # required, "train" / "val" / "test"
  format: # required, "nifti", "h5"
  type: "grouped" # required, "paired" / "unpaired" / "grouped"
  labeled: true # required, true if labeled else false
  moving_image_shape: # required if paired, [dim1, dim2, dim3]
  fixed_image_shape: # required if paired, [dim1, dim2, dim3]
  image_shape: # required if unpaired or grouped, [dim1, dim2, dim3]
  intra_group_prob: 1 # required if grouped, value between [0,1], 0 means inter-group only and 1 means intra-group only
  intra_group_option: "unconstrained" # required if grouped, feasible values are: "forward", "backward", "unconstrained"
  intra_group_enumerate: false # required if grouped, setting true generates all possible data pairs and intra_group_prob must be 1
```
