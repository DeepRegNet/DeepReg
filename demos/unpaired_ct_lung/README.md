# Unpaired CT Lung Registration

Images from different patients often need to be registered for various purposes, for
example to observe effects of regional radiation dose. 

This demo uses the DeepReg toolbox to create a deep learning based image registration 
pipeline where the training data comes from an open source dataset.

# The Dataset

The dataset for this demo comes from [1] and can be downloaded from:
https://zenodo.org/record/3835682#.XsUWXsBpFhE

# Demo Description
This is an unpaired inter-subject registration. Here registeration of expiration 
images with other expiration images from different patients is demonstrated.

# Usage of the Python Scripts and Config Files (applicable to both demos)

The python scripts used along with their brief descriptions are as follows:

- data.py: used to download, unzip and restructure and rescale the dataset to suit 
the needs of the DeepReg toolbox
- .yaml files: config file used to specify training options
- train.py: used to train the network using the DeepReg toolbox
- predict.py: used to predict and visualise predictions from the trained network

# References

[1] Hering, Alessa, Murphy,Keelin, and van Ginneken, Bram. (2020). Lean2Reg Challenge:
CT Lung Registration - Training Data [Data set]. Zenodo.
http://doi.org/10.5281/zenodo.3835682
