# MR data

> Integrated from https://weisslab.cs.ucl.ac.uk/QianyeYang/longi-reg


## H5

The segmentation data and landmark data are stored in separate files.

### Segmentation data

The H5 file has keys in the format of `Patient%d-Visit%d`, 
e.g. `Patient1-Visit2` represents the visit of vID=2 of the patient of pID=1. 
For the same patient, the visit IDs are sorted chronologically but the vID is not necessarily started from 0. 

### Landmarks

The H5 file has keys in the format of `Patient%d-Visit%d-ldmark-%d`, 
e.g. `Patient1-Visit2-ldmark-0` represents the landmark of lID=0 for the visit of vID=2 of the patient of pID=1. 

Same landmark ID always represents the same landmark type.