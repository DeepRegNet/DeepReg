# Data sampling options

## Training, validation and test
Training and validation use the same data folder structure and prediction can use a different test data folder structure. See the [predefined data loader tutorial] (./predefined_loader.md) for more details. 

## Image pair sampling
The registration network takes a pair of images as input.

### Paired-image loader
Using paired-image loader, each pair will be sampled once in each epoch. One epoch iterats over the number of total pairs.

### Unpaired-image loader
Using unpaired-image loader, all the images will be randomly paired first without replacement, before each pair is sampled once in each epoch. One epoch iterates over "floor(number-of-images / 2)" image pairs.

### Grouped-image loader
Using grouped-image loader, each group is sampled once in each epoch. One epoch iterates over the number of groups. 

#### Sampling intra-group image pairs
Each group has multiple intra-group images, one intra-group image pair is randomly sampled (default for training) with different sampling contraints:
    - a) moving image always has a smaller index, e.g. at an earlier time;  
    - b) moving image always has a larger index, e.g. at a later time; or  
    - c) no constraint on the order.  
For the first two options, the intra-subject images will be ascending-sorted by name to represent ordered sequential images, such as time-series data.

#### Option for mixing inter-group image pairs
This option requires an extra parameter specifying the sampling ratio r=[0,1] for intra-to-inter image pairs. When the ratio is greater than zero, there will be rx100% chance to sample the fixed images from a different group, after sampling the moving image from the current intra-group images.

#### Option for iterating all available intra-group image pairs
This option is default for testing, where all the possible image pairs (depends on the intra-group sampling option) will be sampled once in each epoch. This option does not support when mixed intra-and-inter group pairs are sampled.


## Sampling multiple labels
When corresponding labels are available and there are multiple types of labels, e.g. the segmentation of different organs in a CT image. once an image pair is sampled, one label pair is randomly sampled (default for training).

### Label types
When using multiple labels, it is the user's responsibility to ensure the labels are ordered, such that the same label_idx (in [width, height, depth, label_idx]) is the same anatomical or pathological structure between the moving and fixed labels.

### Option for iterating all available label pairs
This option is default for testing, where all the label pairs will be sampled once in each epoch. This option does not support when mixed intra-and-inter group pairs are sampled.

### Inconsistent label types
Inconsistent label types, the numbers of available labels are different between a pair of moving and fixed labels, is not supported in the following two scanarios:
    - Using unpaired image loader;
    - Using grouped-image loader with the mixing inter-group image option enabled.