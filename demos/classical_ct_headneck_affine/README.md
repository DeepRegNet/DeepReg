# Classical Affine Registration for Head-and-Neck CT Images

> **Note**: Please read the
> [DeepReg Demo Disclaimer](https://deepreg.readthedocs.io/en/325-improve-contributing-pages/demo/introduction.html#demo-disclaimer).

This is a special demo that uses the DeepReg package for classical affine image
registration, which iteratively solves an optimisation problem. Gradient descent is used
to minimise the image dissimilarity function of a given pair of moving anf fixed images.

## Author

DeepReg Development Team (raise an issue:
https://github.com/DeepRegNet/DeepReg/issues/new, or mailto the author:
yipeng.hu@ucl.ac.uk)

## Application

Although, in this demo, the moving images are simulated using a randomly generated
transformation. The registration technique can be used in radiotherapy to compensate the
difference between CT acquired at different time points, such as pre-treatment and
intra-/post-treatment.

## Data

https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT

## Instructions

- Change current directory to the root directory of DeepReg project;
- Run `demo_data.py` script to download an example CT volumes with two labels;

```bash
python demos/classical_ct_headneck_affine/demo_data.py
```

- Run `demo_register.py` script. This script will register two images. The fixed image
  will be the downloaded data and the moving image will be simulated by applying a
  random affine transformation, such that the ground-truth is available for. The
  optimised transformation will be applied to the moving images, as well as the moving
  labels. The results, saved in a timestamped folder under the project directory, will
  compare the warped image/labels with the ground-truth image/labels.

```bash
python demos/classical_ct_headneck_affine/demo_register.py
```

## Tested DeepReg Version

v0.1.4

## References

[1] Vallières, M. et al. Radiomics strategies for risk assessment of tumour failure in
head-and-neck cancer. Sci Rep 7, 10117 (2017). doi: 10.1038/s41598-017-10371-5
