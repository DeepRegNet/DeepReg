# Classical Non-rigid Registration for Prostate MR Images

This is a special demo that uses the DeepReg package for claissical nonrigid image
registration, which iteratively solves an optimisation problem. Gradient descent is used
to minimise the image dissimilarity function of a given pair of moving anf fixed images,
often regularised by a deformation smoothness function.

## Author

Yipeng Hu

yipeng.hu@ucl.ac.uk

## Application

Registering inter-subject prostate MR images may be useful to align different glands in
a common space for investigating the spatial distribution of cancer.

## Instructions

- Change current directory to the root directory of DeepReg project;
- Run `demo_data.py` script to download an example MR volumes with the prostate gland
  segmenation;

```bash
python demos/classical_mr_prostate_nonrigid/demo_data.py
```

- Run `demo_register.py` script. This script will register two images. The optimised
  transformation will be applied to the moving images, as well as the moving labels. The
  results will be plotted to compare the warped image/labels with the ground-truth
  image/labels.

```bash
python demos/classical_mr_prostate_nonrigid/demo_register.py
```

## Data

https://promise12.grand-challenge.org/

## Tested DeepReg Version

v0.1.4

## References

[1] Litjens, G., Toth, R., van de Ven, W., Hoeks, C., Kerkstra, S., van Ginneken, B.,
Vincent, G., Guillard, G., Birbeck, N., Zhang, J. and Strand, R., 2014. Evaluation of
prostate segmentation algorithms for MRI: the PROMISE12 challenge. Medical image
analysis, 18(2), pp.359-373.
