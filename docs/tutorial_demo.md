# DeepReg Demos

A typical workflow to develop a [registration network](tutorial_registration.md) using
`DeepReg` includes:

- Select a dataset loader, among the unpaired, paired and grouped, and preppare data
  into folders, as specified in [supported dataset loader details](doc_data_loader.md);
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

## paired_ct_lung

(under development)

## unpaired_ct_lung

(under development)

## DeepReg Demos Index

### [paired_ct_lung](./paired_ct_lung)

#### Paired CT Lung Registration

#### Computed tomography

### [unpaired_ct_lung](./unpaired_ct_lung)

#### Unpaired CT Lung Registration

#### Computed tomography
