# Data sampling options in predefined loaders

This describes the sampling methods used in the [predefined data loaders](./predefined_loader.md). 

## 1 - Image pair sampling
The registration network samples a pair of images as input.

### Paired-image loader
Using paired-image loader, each pair will be sampled once in each epoch. One epoch iterats over the number of total pairs.

### Unpaired-image loader
Using unpaired-image loader, all the images will be randomly paired first without replacement, before each pair is sampled once in each epoch. One epoch iterates over "floor(number-of-images / 2)" image pairs.

### Grouped-image loader
Using grouped-image loader, each group is sampled once in each epoch. One epoch iterates over the number of groups. 

#### Sampling intra-group image pairs
When each group has multiple images, one intra-group image pair is randomly sampled (default for training) with different sampling contraints, by setting `intra_group_option`:  
    - Forward: moving image always has a smaller index, e.g. at an earlier time;  
    - Backward: moving image always has a larger index, e.g. at a later time; or  
    - Unconstrained: no constraint on the order.  
For the first two options, the intra-subject images will be ascending-sorted by name to represent ordered sequential images, such as time-series data.

#### Option for mixing inter-group image pairs
This option requires a parameter `intra_group_prob` specifying the intra-groupe image sampling probability `p=[0,1]`. When `p` is greater than zero, there will be `(1-p)*100%` chance to sample the fixed images from a different group, after sampling the moving image from the current intra-group images.

#### Option for iterating all available intra-group image pairs
All the possible image pairs (depends on the intra-group sampling option) will be sampled once in each epoch. This option is not supported when mixing intra-and-inter-group pairs. Disabling the `sample_image_in_group` will enable this option (default for testing).


## 2 - Label pair sampling
When each image has multiple labels, e.g. segmentations of different organs in a CT image. For each sampled image pair, one label pair is randomly sampled. This is default for training.

### Corresponding label pairs
When using multiple labels, it is the user's responsibility to ensure the labels are ordered, such that the same `label_idx` in `[width, height, depth, label_idx]` is the same anatomical or pathological structure - a corresponding label pair between the moving and fixed labels.  

### Consistent label pairs
Consistent label pairs between a pair of moving and fixed labels requires:  
    1) The two images have the same number of labels; and  
    2) They are ordered and corresponding label types.

When a pair of moving and fixed images have inconsistent label pairs, label dissimilarity can not be defined. Therefore,  
    - Using unpaired-labelled-image loader, consistent label pairs are required;  
    - Using grouped-labelled-image loader, consistent label pairs are required between intra-group image pairs;  
    - When mixing intra-inter-group images in grouped-labelled-image loader, consistent label pairs are required between all intra-and-inter-group image pairs.  
However,  
    - Using paired-labelled-image loader, consistent label pairs are not required between different image pairs;  
    - Using grouped-labelled-image loader without mixing intra-inter-group images, consistent label pairs are not required between different image groups.  


### Option for iterating all available label pairs
This option is default for testing. All the label pairs will be sampled once for each sampled image pair. This option is not supported when mixing intra-and-inter-group image pairs.
