# Add a DeepReg Demo

The [demos](https://github.com/DeepRegNet/DeepReg/tree/main/demos) folder directly under
the DeepReg root directory contains demonstrations using DeepReg for different image
registration applications.

Contributions are welcome! Below is a set of requirements for a demo to be included as a
DeepReg Demo.

- Each demo _must_ have an independent folder directly under `demos/`;
- Name the folder as
  `[loader-type]_[image-modality]_[organ-disease]_[optional:brief-remark]`, e.g.
  `unpaired_ultrasound_prostate` or `grouped_mr_brain_longitudinal`;
- For simplicity, avoid sub-folders (other than those specified below) and separate
  files for additional functions/classes;
- Experiment using cross-validation or advanced data set sampling is NOT encouraged,
  unless the purpose of the demo is to demonstrate how to design experiments.

## Open accessible data

- Each demo _must_ have a `demo_data.py` script to automatically download and preprocess
  demo data;
- Data should be downloaded under the demo folder named `dataset`;
- Data should be hosted in a reliable and efficient (DeepReg repo will not store demo
  data or model) online storage, Kaggle, GitHub and Zendoo are all options for non-login
  access (avoid google drive for known accessibility issues);
- Relevant dataset folder structure to utilise the supported loaders can be either
  pre-arranged in data source or scripted in `demo_data.py` after downloading;
- Avoid slow and excessively large data set download. Consider downloading a subset as
  default for demonstration purpose, with options for full data set.

## Pre-trained model

- A pre-trained model _must_ be available for downloading, with
  github.com/DeepRegNet/deepreg-model-zoo being preferred for storing the models. Please
  contact the Development Team for access;
- The pre-trained model, e.g. ckpt files, should be downloaded and extracted under the
  `dataset` folder. Avoid overwriting with user-trained models;

## Training

- Each demo _must_ have a `demo_train.py` script. If using command line interface
  `deepreg_train`, this file needs to print a message to direct the user to the
  readme.md file (described below) for instructions;
- This is accompanied by one or more config yaml files in the same folder. If
  appropriate, please use the same demo folder name for the config file. Add postfix if
  using multiple config files, e.g. `unpaired_lung_ct_dataset.yaml`,
  `unpaired_lung_ct_train.yaml`.

## Predicting

- Each demo _must_ have a `demo_predict.py` script; If using command line interface
  `deepreg_predict`, this file needs to print a message to direct users to the readme.md
  file (described below) for instructions;
- By default, the pre-trained model should be used in `demo_predict.py`. However, the
  instruction should be clearly given to use the user-trained model, saved with the
  `demo_train.py`;
- Report registration results. Provide at least one piece of numerical metric (Dice,
  distance error, etc) to show the efficacy of the registration. Optimum performance is
  not required;
- Provide at least one piece of visualisation of the results, such as moving image vs
  fixed image vs warped moving image (pred_fixed_image). This may be simply done by
  selecting the typical results from the predict output. If possible, save the
  visualisation to (e.g. png/jpg) files, avoiding compatibility issues. Pointing to the
  relevant file paths generated using `deepreg_predict` is recommended.

## A README.md file

The markdown file _must_ be provided as an entry point for each demo, which should be
based on the [template](../demo/readme_template.html).

Following is a checklist for modifying the README template:

- Modify the link to source code;
- Modify the author section;
- Modify the application section;
- Modify the data section;
- Modify the steps in instruction section;
- Modify the pre-trained model section;
- Modify the tested version;
- Modify the reference section.
