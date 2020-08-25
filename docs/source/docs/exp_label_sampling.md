# Label Sampling

When each image has multiple labels, e.g. segmentations of different organs in a CT
image. For each sampled image pair, one label pair is randomly sampled. This is default
for training.

## Corresponding label pairs

When using multiple labels, it is the user's responsibility to ensure the labels are
ordered, such that the same `label_idx` in `[width, height, depth, label_idx]` is the
same anatomical or pathological structure - a corresponding label pair between the
moving and fixed labels.

## Consistent label pairs

Consistent label pairs between a pair of moving and fixed labels requires: 1) The two
images have the same number of labels; and 2) They are ordered and corresponding label
types.

When a pair of moving and fixed images have inconsistent label pairs, label
dissimilarity can not be defined. Therefore, - Using unpaired-labelled-image loader,
consistent label pairs are required; - Using grouped-labelled-image loader, consistent
label pairs are required between intra-group image pairs; - When mixing
intra-inter-group images in grouped-labelled-image loader, consistent label pairs are
required between all intra-and-inter-group image pairs. However, - Using
paired-labelled-image loader, consistent label pairs are not required between different
image pairs; - Using grouped-labelled-image loader without mixing intra-inter-group
images, consistent label pairs are not required between different image groups.

## Partially labelled image data

When one of the label dissimilarity measures is specified with a nonzero weight, image
data without any label will cause an error in DeepReg. This is a check in place for
avoiding accidentally missing labels. When appropriate, enabling training with missing
labels with a placeholder all-zero mask for the imaging data.

## Option for iterating all available label pairs

This option is default for testing. All the label pairs will be sampled once for each
sampled image pair. This option is not supported when mixing intra-and-inter-group image
pairs.
