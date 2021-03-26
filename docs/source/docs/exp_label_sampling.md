# Label sampling

Images may have multiple labels, such as with segmentation of different organs in CT
scans. In this case, for each sampled image pair, one label pair is randomly chosen by
default.

## Corresponding label pairs

When using multiple labels, ensure the labels are ordered correctly. `label_idx` in
`[width, height, depth, label_idx]` must be the same anatomical or pathological
structure; a corresponding label pair between the moving and fixed labels.

## Consistent label pairs

Consistent label pairs between a pair of moving and fixed labels requires:

1. The two images have the same number of labels, and
2. The labels have the same order

When a pair of moving and fixed images have inconsistent label pairs, label
dissimilarity cannot be defined. The following applies:

- When using the unpaired-labeled-image loader, consistent label pairs are required;
- When using the grouped-labeled-image loader, consistent label pairs are required
  between intra-group image pairs;
- When mixing intra-inter-group images in the grouped-labeled-image loader, consistent
  label pairs are required between all intra-group and inter-group image pairs.

However,

- When using the paired-labeled-image loader, consistent label pairs are not required
  between different image pairs;
- When using the grouped-labeled-image loader without mixing intra-group and inter-group
  images, consistent label pairs are not required between different image groups.

## Partially labeled image data or missing labels

When one of the label dissimilarity measures prevents accidentally missing labels. When
appropriate, enable training with missing labels with placeholder all-zero masks for the
labels.

## Option for iterating all available label pairs

This option is default for testing. All the label pairs will be sampled once for each
sampled image pair. This option is not supported when mixing intra-group and inter-group
image pairs.
