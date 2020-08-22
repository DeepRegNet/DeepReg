DeepReg Demo
============

A typical workflow to develop a `registration network`_ using DeepReg
includes:

- Select a dataset loader, among the `unpaired, paired and grouped`_,
  and prepare data into folders as required;
- Configure the network training in the configuration yaml file(s), as
  specified in `supported configuration details`_;
- Train and tune the registration network with the `command line tool`_
  ``deepreg_train``;
- Test or use the final registration network with the `command line tool`_
  ``deepreg_predict``.

DeepReg has been tested with a wide range of applications with
real-world clinical image and label data. DeepReg Demos all consisted of
open-accessible data sets, step-by-step instructions, pre-trained models
and numerical-and-graphical inference results for demonstration
purposes. These applications range from ultrasound, CT and MR images,
covering many clinical specialities such as neurology, urology,
gastroenterology, oncology, respiratory and cardiovascular diseases.

In particular, the built-in dataset loaders, supporting nifti and h5
file format, provide a variety of training strategies often encountered
in real clinical scenarios, whether images are paired, grouped or
labelled.

This tutorial describe several examples in the DeepReg Demos to explain
how these different scenarios can be implemented with DeepReg.

.. _registration network: tutorial/registration.html
.. _unpaired, paired and grouped: docs/dataset_loader.html
.. _supported configuration details: docs/configuration.html
.. _command line tool: docs/cli.html
