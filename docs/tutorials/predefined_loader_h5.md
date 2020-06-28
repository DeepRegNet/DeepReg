# Data folder structure for h5 images

In the following, we take train directory as an example to list how the files should be placed.

## H5 Data Format

Each `.h5` file is similar to a dictionary, having multiple key-value pairs. Hierarchical multi-level h5 indexing is not used. Each value is either an image volume or a label volume.

### Unpaired images

For unpaired images and labels, each key in the h5 files corresponds to one image or label, e.g. `{"obs1": data1, "obs2": data2, ...}`. The key can be any string such as `obs%d` used here as an example.

#### H5 Case 1-1 Images only

- train
  - images.h5 (keys = ["obs1", "obs2", ...])

#### H5 Case 1-2 Images with labels

- train
  - images.h5 (keys = ["obs1", "obs2", ...])
  - labels.h5 (keys = ["obs1", "obs2", ...])

### Grouped unpaired images

Similar to case 1-1 above, but the keys, in this case, require a specific format with _mandatory_ key string `group-%d-%d`, where `%d` represents a number. For instance, `group-3-2` corresponds to the second observation from the third group, e.g.

#### H5 Case 2-1 Images only

- train
  - images.h5 (keys = ["group-1-1", "group-1-2", "group-2-1", ...])

#### H5 Case 2-2 Images with labels

- train
  - images.h5 (keys = ["group-1-1", "group-1-2", "group-2-1", ...])
  - labels.h5 (keys = ["group-1-1", "group-1-2", "group-2-1", ...])

### Paired images

Paired data are required to be placed in files, with _specific file names_, `train/moving_images.h5`, `train/fixed_images.h5`, `train/moving_labels.h5`, and `train/fixed_labels.h5`.

#### H5 Case 3-1 Images only

- train
  - moving_images.h5 (keys = ["obs1", "obs2", ...])
  - fixed_images.h5 (keys = ["obs1", "obs2", ...])

#### H5 Case 3-2 Images with labels

- train
  - moving_images.h5 (keys = ["obs1", "obs2", ...])
  - fixed_images.h5 (keys = ["obs1", "obs2", ...])
  - moving_labels.h5 (keys = ["obs1", "obs2", ...])
  - fixed_labels.h5 (keys = ["obs1", "obs2", ...])
