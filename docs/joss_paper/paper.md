---
title: 'DeepReg: a deep-learning toolkit for medical image registration'
tags:
  - Python
  - TensorFlow
  - medical image registration
  - image fusion
  - deep learning
  - neural networks
authors:  # not-ordered yet
  - name: Yunguan Fu
    orcid: 0000-0002-1184-7421
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Nina Montana Brown
    orcid: 0000-0001-5685-971X
    affiliation: "1, 2"
  - name: Shaheer Saeed
    orcid: 0000-0002-5004-0663
    affiliation: "1, 2"
  - name: Adrià Casamitjana
    affiliation: 2
  - name: Zachary Baum
    orcid: 0000-0001-6838-335X
  - name: Rémi Delaunay
    affiliation: "1, 4"
  - name: Qianye Yang
    orcid: 0000-0003-4401-5311
    affiliation: "1, 2"
  - name: Alexander Grimwood
    affiliation: "1, 2"
  - name: Zhe Min
    affiliation: 1
  - name: Ester Bonmati
    affiliation: "1, 2"
  - name: Matthew J. Clarkson
    affiliation: "1, 2"
  - name: Tom Vercauteren
    orcid: 0000-0003-1794-0456
    affiliation: 4
  - name: Yipeng Hu
    orcid: 0000-0003-4902-0486
    affiliation: "1, 2"
affiliations:
 - name: Wellcome/EPSRC Centre for Surgical and Interventional Sciences, University College London
   index: 1
 - name: Centre for Medical Image Computing, University College London
   index: 2
 - name: InstaDeep
   index: 3
 - name: Department of Surgical & Interventional Engineering, King’s College London
   index: 4
date: 12 August 2020
bibliography: paper.bib
---

# Summary
Image fusion is a fundamental task in medical image analysis and computer assisted intervetnion. Medical image registration, a class of algorithms to align different images together, has in recent years turned the research attention towards deep learning. Indeed, the representation ability to learn from population data with deep neural networks opened new possibilities for accelerated registration excuation during inference, improved registration accuracy and more robust generalisation that is less dependent on hand-engineered image features and similarity measures, for feature-based- and intensity-based image registration algorithms, respectively.

`DeepReg` is a Python package that implements a class of image registration algorithms based on deep neural networks. Both unsupervised algorithms [@de2019deep] and weakly-supervised algortihms [@hu2018label] have been implemented. In addition, combining unsupervised losses, such as those derived from image similarity measures, and weak supervision, such as overlap between corresponding anatomical and pathological regions, is at the core of this framework. In this paper, we outline the algorithms and several representative clinical applications, available as 'demos' at the time of writing. A summary of data input and output is also provided to describe the implemented dataset loaders and training data sampling methods. Collectively, we demonstrate the versatility of the package, by providing a wide range of clinical applications as example demonstrations in the `demos` folder.

`DeepReg` provides a command line tool option that implements basic functionalities including model training, prediction and image warping, with options and configurations being specified through the command line arguments and/or configuration files in .yaml format.

# Algorithms
## Unsupervised learning
Unsupervised learning was first developed independently from a number of research groups, among which, [@de2019deep] demonstrated the applicability in real clinical applications while VoxelMorph [@balakrishnan2019voxelmorph] has contributed to a popular open-source repository for several algorithms tested using brain MR images. Image dissimilarity is measured between the fixed and warped moving images, which has been adapted from the classical image registration methods. Fig.\autoref{fig:unsupervised} shows a schematic illustration of the network training. Image dissimilarity measures include sum-of-square difference in intensity (SSD), normalised cross-correlation (NCC), mutual information (MI) and their variants.

![Registration network with unsupervised loss.\label{fig:unsupervised}](../asset/deepreg-tutorial-unsupervised.svg)

## Weakly-supervised learning
Weak supervision utilising segmented corresponding regions in the medical image pairs was first proposed in a multimodal application for registering prostate MR to ultrasound images [@hu2018label]. In addition to the regularisation on the predicted displacement field, the training is driven by minimising the dissimilarity between the fixed labels and the warped moving labels, as illustrated in Fig.X. This formulation is modality-independent and similar to many other well-studied computer vision and medical imaging tasks, such as image segmentation. When multiple corresponding label pairs available, a two-stage sampling strategy can be adopted for an unbiased gradient back-propagation during training. Label dissimilarity measures include Dice, Jaccard, cross-entropy and their variants, such as a multiscale Dice.

![Registration network with weak supervision loss.\label{fig:weakly}](../asset/deepreg-tutorial-weakly.svg)

## Combining unsupervised loss with weak supervision
Combining the unsupervised loss and the weak supervision has shown superior registration accuracy, compared with that using unsupervised loss alone [@balakrishnan2019voxelmorph]. As a result, the overall loss is the weighted sum of the image-, label dissimilarity and deformation regularisation.

![Registration network with combined unsupervised and weakly-supervised losses.\label{fig:combined}](../asset/deepreg-tutorial-combined.svg)

## Deformation regularisation and conditional segmentation
The loss functions described in the previous sections are often combined with a deformation regularisation term on the predicted displacement field, in order to ensure the predicted deformation is smooth. The deformation can be regularised by L1-, L2 norms of the displacement gradient and bending energy, based on the first- or second dirivatives of DDF.

The latest result from a different registration formulation challenged the use of deformation regularisation and consequently the necessity of predicting dense displacement field [@hu2019conditional], also implemented in DeepReg.

# Implemetation
## Model
A pair of moving image and fixed image is the input of the network. When available, pairs of moving labels and/or fixed labels can also be the input of the network, regardless of whether they are used to compute the loss. The network predicts a dense displacement field (DDF). Options are also available for the DDF being integrated by a predicted dense velocity field (DFV) or computed from a predicted affine transformation. For conditional segmentation, the network directly predicts warped moving labels [@hu2019conditional].

An encoder-decoder architecture, such as an adapted U-Net and the local-net [@hu2018weakly] is used for dense prediction such as direct DDF or DVF. An encoder-only architecture, such as ResNet and VGG, is used to predict the parameters of an affine transformation model, with 12 degrees of freedom.

## Model training
Registration networks are trained using standard minibatch gradient descent, with each minibatch sampling pairs of moving-fixed images and, when available, moving-and-fixed labels.

`DeepReg`, at its first release, provides six predefined dataset loaders for three types of image pairs, unpaired, grouped and paired, each with two dataset loaders depends on whether corresponding labels are avaialble. Random sampling methods are used for unpaired images. For grouped images, group will be sampled before randomly sampling intra-group images for train without biasing to different groups. When multiple labels are avaialble associated with each image, another staged sampling methods are used to draw one pair of labels for each sampled image pair, during each training iteration. Details of the available dataset loaders, how to add cutomised loader and the sampling options are provided in a series of tutorials under the `tutorials` folder.

## The software
`DeepReg` is unit tested, and includes both technical documentation and various tutorials to introduce users to various use cases using open-sourced datasets. The continuous integration platform TravisCI (https://travis-ci.org/) is used together with a cutomised contributor guideline to facilitate contributions.


# Example applications
`DeepReg` provides a collection of demonstrations, _DeepReg Demos_, using open-accesible data with real clinical applications.

## Paired images
If images are available in pairs, two dataset loaders are currently available for paired images with- and without labels. Many clinical application for tracking organ motion and other temporal changes between two time points require _intra-subject_ _single-modality_ image registration. Registering lung CT images for the same patient, acquired at expritory and inspritory, is such an example of both unsupervised (without labels) and combined supervision (with segmentation of anatomical structures). Furthermore, registering prostate MR, aquired before surgery, and the intra-operative trasnrectal ultrasound images is an example of weakly-supervised training for multimodal image registration.

## Unpaired images
Unpaired images are found in applications such as single-modality inter-subject registration. Two dataset loaders for labelled and unlabelled image data. Registering different MR images of brain from different subjects has played a fundamental role in population studies in neuroimaging. An example registering 3D ultrasound images from different surgical patients are also provided, especially to demonstrate the support for experiments such as cross-validation in DeepReg.

## Grouped images
Unpaired images can also be grouped in applications such as single-modality intra-subject registration, in which each subject has multiple images acquired, e.g. at different time points. Two dataset loaders are implemented for grouped image data, with- and without labels. Paired dataset loader is a special case of the grouped dataset loader. Prostate logitudinal MR is proposed to track the progression of prostate cancer patients at an early stage during the active surveilence programme [@yang2020longitudinal]. One DeepReg demo illustrates how the grouped dataset loader can be used for deep-learning based feature reigstration by aligning intra-patient organ masks.


# Conclusion
`DeepReg` facilitates a number of deep learning algorithms to train image registration networks, which provides a reference of basic functionality and performance to further research in fields such as medical image analysis and computer assisted intervention. It is in its current open-source format not only providing a tool for scientific research, but also welcoming contributions from wider communities above and beyond medical imaging, biomedical engineering and computer sicence.

# Acknowledgements

This work is supported by the Wellcome/EPSRC Centre for Interventional and Surgical Sciences (203145Z/16/Z). Support was also from the Engineering and Physical Sciences Research Council (EPSRC) (NS/A000049/1) and Wellcome Trust (203148/Z/16/Z). TV is supported by a Medtronic / Royal Academy of Engineering Research Chair (RCSRF1819\7\34).

# References
<!-- This will be filled in by references in paper.bib -->
