# DeepReg demo

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
- Data for training and test should be downloaded under the demo folder named `dataset`,
  such as `dataset/train` and `dataset/test`;
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
  `dataset/pretrained` folder. Avoid overwriting with user-trained models;

## Training

- Each demo _must_ have a `demo_train.py` script;
- This is accompanied by one or more config yaml files in the same folder. Please use
  the same demo folder name for the config file. Add postfix if multiple training
  methods are provided, e.g. `unpaired_ct_abdomen_comb.yaml`,
  `unpaired_ct_abdomen_unsup.yaml`.

## Predicting

- Each demo _must_ have a `demo_predict.py` script;
- By default, the pre-trained model should be used in `demo_predict.py`. However, the
  instruction should be clearly given to use the user-trained model, saved with the
  `demo_train.py`;

## A README.md file

A markdown file _must_ be provided under `demos/<demo_name>` as an entry point for each
demo, which should be based on the [template](../demo/readme_template.html). Moreover, a
`.rst` file _must_ be provided under `docs/source/demo` to link the markdown file to the
documentation page. The
[introduction.rst](https://github.com/DeepRegNet/DeepReg/blob/main/docs/source/demo/introduction.rst)
file should be updated properly as well.

Following is a checklist for modifying the README template:

- Update the link to source code;
- Update the author section;
- Update the application section;
- Update the data section, optionally, describe the used pre-processing methods;
- Update the `name` in all commands;
- Update the reference section.
- Optionally, adapt the file to custom needs.
