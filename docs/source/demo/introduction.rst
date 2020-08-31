Introduction to DeepReg Demos
=============================

DeepReg offers multiple built-in dataset loaders to support real-world clinical scenarios, in which images may be paired, unpaired or grouped. Images may also be labeled with segmented regions of interest to assist registration.

A typical workflow to develop a `registration network`_ using DeepReg
includes:

- Select a dataset loader, among the `unpaired, paired and grouped`_,
  and prepare data into folders as required;
- Configure the network training in the configuration yaml file(s), as
  specified in `supported configuration details`_;
- Train and tune the registration network with the `command line tool`_
  ``deepreg_train``;
- Test or use the final trained registration network with the `command line tool`_
  ``deepreg_predict``.

Besides the tutorials, a series of DeepReg Demos are provided to
showcase a wide range of applications with real clinical image and label data.
These applications range from ultrasound, CT and MR images,
covering many clinical specialties such as neurology, urology,
gastroenterology, oncology, respiratory and cardiovascular diseases.

Each DeepReg Demo provides a step-by-step instruction
to explain how different scenarios can be implemented with DeepReg.
All data sets used are open-accessible.
Pre-trained models with numerical and graphical inference results are also available.

.. _demo-disclaimer:

.. note::

   DeepReg Demos are provided to demonstrate functionalities in DeepReg.
   Although effort has been made to ensure these demos are representative
   of real-world applications, the implementations and the results are not
   peer-reviewed or tested for clinical efficacy. Substantial further
   adaptation and development may be required for any potential clinical
   adoption.

.. _registration network: ../tutorial/registration.html
.. _unpaired, paired and grouped: ../docs/dataset_loader.html
.. _supported configuration details: ../docs/configuration.html
.. _command line tool: ../docs/cli.html

Paired Images
=============

The following DeepReg Demos provide examples of
using paired images.

- `Paired lung CT registration <paired_ct_lung.html>`__

  This demo registers paired CT lung images, with optional weak supervision.

- `Paired brain MR-ultrasound registration <paired_mrus_brain.html>`__

  This demo registers paired preoperative MR images and 3D tracked ultrasound images for
  locating brain tumours during neurosurgery, with optional weak supervision.

- `Paired prostate MR-ultrasound registration <paired_mrus_prostate.html>`__

  This demo registers paired MR-to-ultrasound prostate images, an example of
  weakly-supervised multimodal image registration.

.. toctree::
    :glob:
    :hidden:
    :maxdepth: 2

    paired_*

Unpaired Images
===============

The following DeepReg Demos provide examples of
using unpaired images.

- `Unpaired abdominal CT registration <unpaired_ct_abdomen.html>`__

  This demo compares three training strategies, using unsupervised, weakly-supervised and
  combined losses, to register inter-subject abdominal CT images.

- `Unpaired lung CT registration <unpaired_ct_lung.html>`__

  This demo registers unpaired CT lung images, with optional weak supervision.

- `Unpaired hippocampus MR registration <unpaired_mr_brain.html>`__

  This demo aligns hippocampus on MR images between different patients, with optional weak
  supervision.

- `Unpaired prostate ultrasound registration <unpaired_us_prostate_cv.html>`__

  This demo registers 3D ultrasound images with a 9-fold cross-validation. This strategy
  is applicable for any of the available dataset loaders.

.. toctree::
    :glob:
    :hidden:
    :maxdepth: 2

    unpaired_*

Grouped Images
==============

The following DeepReg Demos provide examples of using grouped images.

- `Pairwise registration for grouped prostate segmentation masks <grouped_mask_prostate_longitudinal.html>`__

  This demo registers grouped masks (as input images) of prostate glands from MR images,
  an example of feature-based registration.

- `Pairwise registration for grouped cardiac MR images <grouped_mr_heart.html>`__

  This demo registers grouped CMR images, where each group has multi-sequence CMR images from a single patient.

.. toctree::
    :glob:
    :hidden:
    :maxdepth: 2

    grouped_*

Classical Registration
======================

The following DeepReg Demos provide examples of
using classical registration methods.

- `Classical affine registration for head-and-neck CT images <classical_ct_headneck_affine.html>`__

  This demo registers head-and-neck CT images using iterative affine registration.

- `Classical nonrigid registration for prostate MR images <classical_mr_prostate_nonrigid.html>`__

  This demo registers prostate MR images using iterative nonrigid registration.

.. toctree::
    :hidden:
    :glob:
    :maxdepth: 2

    classical_*
