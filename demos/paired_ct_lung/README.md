# Paired CT Lung Registration

Lungs experince motion due to breathing and the motion can be a problem when registering
two images of the lung taken at different times in the breathing cycle. This demo uses
the DeepReg toolbox to create a deep learning based image registration pipeline where
the training data comes from an open source dataset.

# The Dataset

The dataset for this demo comes from [1] and can be downloaded from:
https://zenodo.org/record/3835682#.XsUWXsBpFhE

# Demo Description
This is a paired intra subject registration. This means registration of inspiration
images with expiration images for the same patient is demonstrated.

# Usage of the Python Scripts and Config File

The python scripts used along with their brief descriptions are as follows:

- demo_data.py: used to download, unzip and restructure the dataset to suit the needs of the
  DeepReg toolbox
- paired_ct_lung.yaml files: config file used to specify training options
- demo_train.py: used to rescale images to 255 and train the network using the DeepReg
  toolbox
- demo_predict.py: used to predict and visualise predictions from the trained network

# References

[1] Hering, Alessa, Murphy,Keelin, and van Ginneken, Bram. (2020). Lean2Reg Challenge:
CT Lung Registration - Training Data [Data set]. Zenodo.
http://doi.org/10.5281/zenodo.3835682
