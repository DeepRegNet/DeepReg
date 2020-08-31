# Classical nonrigid registration for prostate MR images

> **Note**: Please read the
> [DeepReg Demo Disclaimer](introduction.html#demo-disclaimer).

[Source Code](https://github.com/DeepRegNet/DeepReg/tree/main/demos/classical_mr_prostate_nonrigid)

This is a special demo that uses the DeepReg package for classical nonrigid image
registration, which iteratively solves an optimisation problem. Gradient descent is used
to minimise the image dissimilarity function of a given pair of moving and fixed images,
often regularised by a deformation smoothness function.

## Author

DeepReg Development Team

## Application

Registering inter-subject prostate MR images may be useful to align different glands in
a common space for investigating the spatial distribution of cancer.

## Data

https://promise12.grand-challenge.org/

## Instruction

- Change current directory to the root directory of DeepReg project;
- Run `demo_data.py` script to download an example MR volumes with the prostate gland
  segmentation;

```bash
python demos/classical_mr_prostate_nonrigid/demo_data.py
```

- Run `demo_register.py` script. This script will register two images. The optimised
  transformation will be applied to the moving images, as well as the moving labels. The
  results, saved in a timestamped folder under the project directory, will compare the
  warped image/labels with the ground-truth image/labels.

```bash
python demos/classical_mr_prostate_nonrigid/demo_register.py
```

## Tested DeepReg version

0.1.0

## Contact

Please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose).

## Reference

[1] Litjens, G., Toth, R., van de Ven, W., Hoeks, C., Kerkstra, S., van Ginneken, B.,
Vincent, G., Guillard, G., Birbeck, N., Zhang, J. and Strand, R., 2014. Evaluation of
prostate segmentation algorithms for MRI: the PROMISE12 challenge. Medical image
analysis, 18(2), pp.359-373.
