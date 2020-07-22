# Classical affine registration for head-and-neck CT images

This is a special demo that demonstrates using the DeepReg package for claissical affine
image registration, which iteratively solves an optimisation problem. Gradient descent
is used to minimise the image dissimilarity function of a given pair of moving anf fixed
images.

## Author

Yipeng Hu (yipeng.hu@ucl.ac.uk)

## Instructions

- Run the demo_data.py script to download an example CT volumes with 2 labels;
- Run the demo_register.py script. This script will register two images. The fixed image
  will be the downloaded data and the moving image will be simulated by applying a
  random affine transformation, such that the ground-truth is available for. The
  optimised transformation will be applied to the moving images, as well as the moving
  labels. The results will be plotted to compare the warped image/labels with the
  ground-truth image/labels.

## Application

Although, in this demo, the moving images are simulated using a randomly generated
transformation. The registration technique can be used in radiotherapy to compensate the
difference between CT acquired at different time points, such as pre-treatment and
intra-/post-treatment.

## Data

https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT

## Tested DeepReg Version

v0.1.4

## References

Vallières, M. et al. Radiomics strategies for risk assessment of tumour failure in
head-and-neck cancer. Sci Rep 7, 10117 (2017). doi: 10.1038/s41598-017-10371-5
