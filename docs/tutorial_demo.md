# DeepReg Demo

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
covering many clinical specialities such as neurology, urology, gastroentrology,
oncology, respirotory and cadiovascular diseases.

In particular, the built-in dataset loaders, supporting nifti and h5 file format,
provide a varity of training strategies often encountered in real clinical scenarios,
whether images are paired, grouped or labelled.

This tutorial describe several examples in the DeepReg Demos to explain how these
different scenarios can be implemented with DeepReg.

## Train with paired images

- [paired_ct_lung](https://github.com/DeepRegNet/DeepReg/tree/master/demos/paired_ct_lung)

This demo registers paired CT lung images, with optional weak supervision.

- [paired_mrus_prostate](https://github.com/DeepRegNet/DeepReg/tree/master/demos/paired_mrus_prostate)

This demo registers paired MR-to-ultrasound prostate images, an example of
weakly-supervised multimodal image registration.

- [paired_mrus_brain](https://github.com/DeepRegNet/DeepReg/tree/master/demos/paired_mrus_brain)

This demo registers paired preoperative MR images and 3D tracked ultrasound images for
locating brain tumours during neurosurgery, with optional weak supervision.

## Train with unpaired images

- [unpaired_ct_abdomen](https://github.com/DeepRegNet/DeepReg/tree/master/demos/unpaired_ct_abdomen)

(under development)

- [unpaired_mr_brain](https://github.com/DeepRegNet/DeepReg/tree/master/demos/unpaired_mr_brain)

This demo aligns hippocampus on MR images between different patients, with optional weak
supervision.

- [unpaired_ct_lung](https://github.com/DeepRegNet/DeepReg/tree/master/demos/unpaired_ct_lung)

This demo registers unpaired CT lung images, with optional weak supervision.

## Train with grouped images

- [grouped_mask_prostate_longitudinal](https://github.com/DeepRegNet/DeepReg/tree/master/demos/grouped_mask_prostate_longitudinal)

This demo registers grouped masks (as input images) of prostate glands from MR images,
an example of feature-based registration.

## Experiment with cross-validation

- [unpaired_us_prostate_cv](https://github.com/DeepRegNet/DeepReg/tree/master/demos/unpaired_us_prostate_cv)

This demo registers 3D ultrasound images with a 9-fold cross-validation. This strategy
is applicable for any of the available dataset loaders.

## Classical image registration

- [classical_ct_headneck_affine](https://github.com/DeepRegNet/DeepReg/tree/master/demos/classical_ct_headneck_affine)

This demo registers head-and-neck CT images using iterative affine registration.

- [classical_mr_prostate_nonrigid](https://github.com/DeepRegNet/DeepReg/tree/master/demos/classical_mr_prostate_nonrigid)

This demo registers prostate MR images using iterative nonrigid registration.
