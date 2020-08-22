Unpaired Images
===============

The following DeepReg demos provide examples of
using unpaired images.

- `Unpaired ultrasound images registration <unpaired_us_prostate_cv.html>`__

  This demo registers 3D ultrasound images with a 9-fold cross-validation. This strategy
  is applicable for any of the available dataset loaders.

- `Unpaired CT abdominal registration <unpaired_ct_abdomen.html>`__

  This demo compares three training strategies, using unsupervised, weakly-supervised and
  combined losses, to register inter-subject abdominal CT images.

- `Unpaired MR hippocampus registration <unpaired_mr_brain.html>`__

  This demo aligns hippocampus on MR images between different patients, with optional weak
  supervision.

- `Unpaired CT lung registration <unpaired_ct_lung.html>`__

  This demo registers unpaired CT lung images, with optional weak supervision.

.. toctree::
    :hidden:
    :maxdepth: 2

    unpaired_us_prostate_cv
    unpaired_ct_abdomen
    unpaired_mr_brain
    unpaired_ct_lung
