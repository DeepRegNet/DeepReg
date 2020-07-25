# DeepReg Demos

A typical workflow to develop a [registration network](tutorial_registration.md) using
`DeepReg` includes:

- Select a dataset loader, among the [unpaired, paired and grouped](doc_data_loader.md),
  and prepare data into folders as required;
- Configure the network training in the configuration yaml file(s), as specified in
  [supported configuration details](doc_configuration.md);
- Train and tune the registration network with command line tool `train`;
- Test or use the final registration network with command line tool `predict`.

`DeepReg` has been tested with a wide range of applications with real-world clinical
image and label data. `DeepReg Demos` all consisted of open-accesible dataset,
step-by-step instructions, pre-trained models and numerical-and-graphical inference
results for demonstration purposes. These applications range from ultrasound, CT and MR
images, covering many clinical specialities such as neurology, urology, gastroentrology,
oncology, respirotory and cadiovescular diseases.

In particular, the built-in dataset loaders, supporting nifti and h5 file format,
provide a varity of training strategies often encountered in real clinical scenarios,
whether images are paired, grouped or labelled.

This tutorial describe several examples in the `DeepReg Demos` to explain how these
different scenarios can be implemented with `DeepReg`. A complete list of demos can be
found in the [DeepReg Demos Index](#deepreg-demos-index)

## Paired image registration

- [paired_ct_lung](https://github.com/DeepRegNet/DeepReg/tree/master/demos/paired_ct_lung)

This demo registers paired CT lung images.

## Unpaired image registration

- [unpaired_ct_lung](https://github.com/DeepRegNet/DeepReg/tree/master/demos/unpaired_ct_lung)

This demo registers unpaired CT lung images.

## Grouped image registration

(under development)

## Experiment with cross-validation

- [unpaired_us_prostate_cv](https://github.com/DeepRegNet/DeepReg/tree/master/demos/unpaired_us_prostate_cv)

This demo registers 3D ultrasound images with a 9-fold cross-validation. This strategy
is applicable for any of the available dataset loaders.

## Classical image registration

- [classical_ct_headandneck_affine](https://github.com/DeepRegNet/DeepReg/tree/master/demos/classical_ct_headandneck_affine)

This demo registers head-and-neck CT images using iterative affine registration.

- [classical_mr_prostate_nonrigid](https://github.com/DeepRegNet/DeepReg/tree/master/demos/classical_mr_prostate_nonrigid)

This demo registers prostate MR images using iterative nonrigid registration.
