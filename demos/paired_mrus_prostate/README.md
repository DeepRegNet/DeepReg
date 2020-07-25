# Paired MR-to-Ultrasound registration - an example of weakly-supervised label-driven training

This demo uses DeepReg to re-implament the algorithms described in
[Weakly-supervised convolutional neural networks for multimodal image registration](https://doi.org/10.1016/j.media.2018.07.002).
A standalone demo was hosted at https://github.com/YipengHu/label-reg.

## Author

Yipeng Hu (yipeng.hu@ucl.ac.uk)

## Application

Registering preoperative MR images to intraoperative transrectal ultrasound images has
been an active research area for more than a decade. The multimodal image registration
task assist a number of ultrasound-guided intervention and surgical procedures, such as
targted biopsy and focal therapy for prostate cancer patients. One of the key challenges
in this registration tasks is the lack of robust and effective similarity measures
between the two image types. This demo implements a weakly-supervised learning approach
to learn voxel correspondence between intensity patterns between the multimodal data,
driven by expert-defined anatomical landmarks, such as the prostate gland segmenation.

## Instructions

## Data

This is a demo without real clinical data. The MR and ultrasound images used are
simulated dummy images.

## Tested DeepReg Tag

0.14
