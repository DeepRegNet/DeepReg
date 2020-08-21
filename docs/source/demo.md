# Demo

A typical workflow to develop a [registration network](tutorial_registration.md) using
DeepReg includes:

- Select a dataset loader, among the [unpaired, paired and grouped](doc_data_loader.md),
  and prepare data into folders as required;
- Configure the network training in the configuration yaml file(s), as specified in
  [supported configuration details](doc_configuration.md);
- Train and tune the registration network with the [command line tool](doc_command.md)
  `deepreg_train`;
- Test or use the final registration network with the
  [command line tool](doc_command.md) `deepreg_predict`.

DeepReg has been tested with a wide range of applications with real-world clinical image
and label data. DeepReg Demos all consisted of open-accessible data sets, step-by-step
instructions, pre-trained models and numerical-and-graphical inference results for
demonstration purposes. These applications range from ultrasound, CT and MR images,
covering many clinical specialities such as neurology, urology, gastroenterology,
oncology, respirotory and cadiovascular diseases.

In particular, the built-in dataset loaders, supporting nifti and h5 file format,
provide a varity of training strategies often encountered in real clinical scenarios,
whether images are paired, grouped or labelled.

This tutorial describe several examples in the DeepReg Demos to explain how these
different scenarios can be implemented with DeepReg.

## Train with paired images

### Paired CT lung registration

This demo registers paired CT lung images, with optional weak supervision.

Check [README](https://github.com/DeepRegNet/DeepReg/tree/master/demos/paired_ct_lung)
for more details.

### Paired MR-US registration

This demo registers paired MR-to-ultrasound prostate images, an example of
weakly-supervised multimodal image registration.

Check
[README](https://github.com/DeepRegNet/DeepReg/tree/master/demos/paired_mrus_prostate)
for more details.

### Paired MR-US brain registration

This demo registers paired preoperative MR images and 3D tracked ultrasound images for
locating brain tumours during neurosurgery, with optional weak supervision.

Check
[README](https://github.com/DeepRegNet/DeepReg/tree/master/demos/paired_mrus_brain) for
more details.

## Train with unpaired images

### Unpaired CT abdominal registration

This demo compares three training strategies, using unsupervised, weakly-supervised and
combined losses, to register inter-subject abdominal CT images.

Check
[README](https://github.com/DeepRegNet/DeepReg/tree/master/demos/unpaired_ct_abdomen)
for more details.

### Unpaired MR hippocampus registration

This demo aligns hippocampus on MR images between different patients, with optional weak
supervision.

Check
[README](https://github.com/DeepRegNet/DeepReg/tree/master/demos/unpaired_mr_brain) for
more details.

### Unpaired CT lung registration

This demo registers unpaired CT lung images, with optional weak supervision.

Check [README](https://github.com/DeepRegNet/DeepReg/tree/master/demos/unpaired_ct_lung)
for more details.

## Train with grouped images

### Pairwise registration for grouped prostate images

This demo registers grouped masks (as input images) of prostate glands from MR images,
an example of feature-based registration.

Check
[README](https://github.com/DeepRegNet/DeepReg/tree/master/demos/grouped_mask_prostate_longitudinal)
for more details.

## Experiment with cross-validation

### Unpaired ultrasound images

This demo registers 3D ultrasound images with a 9-fold cross-validation. This strategy
is applicable for any of the available dataset loaders.

Check
[README](https://github.com/DeepRegNet/DeepReg/tree/master/demos/unpaired_us_prostate_cv)
for more details.

## Classical image registration

### Classical affine registration for head-and-neck CT images

This demo registers head-and-neck CT images using iterative affine registration.

Check
[README](https://github.com/DeepRegNet/DeepReg/tree/master/demos/classical_ct_headneck_affine)
for more details.

### Classical nonrigid registration for prostate MR images

This demo registers prostate MR images using iterative nonrigid registration.

Check
[README](https://github.com/DeepRegNet/DeepReg/tree/master/demos/classical_mr_prostate_nonrigid)
for more details.
