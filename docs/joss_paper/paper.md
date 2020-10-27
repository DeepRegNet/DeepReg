---
title: 'DeepReg: a deep learning toolkit for medical image registration'
tags:
  - Python
  - TensorFlow
  - medical image registration
  - image fusion
  - deep learning
  - neural networks
authors:
  - name: Yunguan Fu
    orcid: 0000-0002-1184-7421
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Nina Montaña Brown
    orcid: 0000-0001-5685-971X
    affiliation: "1, 2"
  - name: Shaheer U. Saeed
    orcid: 0000-0002-5004-0663
    affiliation: "1, 2"
  - name: Adrià Casamitjana
    orcid: 0000-0002-0539-3638
    affiliation: 2
  - name: Zachary M. C. Baum
    affiliation: "1, 2"
    orcid: 0000-0001-6838-335X
  - name: Rémi Delaunay
    orcid: 0000-0002-0398-4995
    affiliation: "1, 4"
  - name: Qianye Yang
    orcid: 0000-0003-4401-5311
    affiliation: "1, 2"
  - name: Alexander Grimwood
    orcid: 0000-0002-2608-2580
    affiliation: "1, 2"
  - name: Zhe Min
    orcid: 0000-0002-8903-1561
    affiliation: 1
  - name: Stefano B. Blumberg
    orcid: 0000-0002-7150-9918
    affiliation: 2
  - name: Juan Eugenio Iglesias
    orcid: 0000-0001-7569-173X
    affiliation: "2, 5, 6"
  - name: Dean C. Barratt
    orcid: 0000-0003-2916-655X
    affiliation: "1, 2"
  - name: Ester Bonmati
    orcid: 0000-0001-9217-5438
    affiliation: "1, 2"
  - name: Daniel C. Alexander
    orcid: 0000-0003-2439-350X
    affiliation: 2
  - name: Matthew J. Clarkson
    orcid: 0000-0002-5565-1252
    affiliation: "1, 2"
  - name: Tom Vercauteren
    orcid: 0000-0003-1794-0456
    affiliation: 4
  - name: Yipeng Hu
    orcid: 0000-0003-4902-0486
    affiliation: "1, 2"
affiliations:
 - name: Wellcome/EPSRC Centre for Surgical and Interventional Sciences, University College London, London, UK
   index: 1
 - name: Centre for Medical Image Computing, University College London, London, UK
   index: 2
 - name: InstaDeep, London, UK
   index: 3
 - name: Department of Surgical & Interventional Engineering, King’s College London, London, UK
   index: 4
 - name: Martinos Center for Biomedical Imaging, Massachusetts General Hospital and Harvard Medical School, Boston, USA
   index: 5
 - name: Computer Science and Artificial Intelligence Laboratory, Massachusetts Institute of Technology, Boston, USA
   index: 6
date: 1 September 2020
bibliography: paper.bib
---

# Summary
Image fusion is a fundamental task in medical image analysis and computer-assisted intervention. Medical image registration, computational algorithms that align different images together [@hill2001medical], has in recent years turned the research attention towards deep learning. Indeed, the representation ability to learn from population data with deep neural networks has opened new possibilities for improving registration generalisability by mitigating difficulties in designing hand-engineered image features and similarity measures for many real-world clinical applications [@haskins2020deep; @fu2020deep]. In addition, its fast inference can substantially accelerate registration execution for time-critical tasks.

*DeepReg* is a Python package using TensorFlow [@tensorflow2015-whitepaper] that implements multiple registration algorithms and a set of predefined _dataset loaders_, supporting both labelled- and unlabelled data. DeepReg also provides command-line tool options that enable basic and advanced functionalities for model training, prediction and image warping. These implementations, together with their documentation, tutorials and demos, aim to simplify workflows for prototyping and developing novel methodology, utilising latest development and accessing quality research advances. DeepReg is unit tested and a set of customised contributor guidelines are provided to facilitate community contributions.

A submission to the MICCAI Educational Challenge has utilised the DeepReg code and demos to explore the link between classical algorithms and deep-learning-based methods [@brown2020introduction], while a recently published research work investigated temporal changes in prostate cancer imaging, by using a longitudinal registration adapted from the DeepReg code [@yang2020longitudinal].

# Statement of need
Currently, popular packages focusing on deep learning methods for medical imaging, such as NiftyNet [@gibson2018niftynet] and MONAI (https://monai.io/), do not support image registration. The existing open-sourced registration projects either implement specific published algorithms without automated testing, such as the VoxelMorph [@balakrishnan2019voxelmorph], or focus on classical methods, such as NiftiReg [@modat2010fast], SimpleElastix [@marstal2016simpleelastix] and AirLab [@sandkuhler2018airlab]. Therefore an open-sourced project focusing on image registration with deep learning is much needed for general research and education purposes.

# Implementation
DeepReg implements a framework for unsupervised learning [@de2019deep; @balakrishnan2019voxelmorph], weakly-supervised learning [@hu2018label; @hu2018weakly] and their combinations and variants, e.g. [@hu2019conditional]. Many options are included for major components of these approaches, such as different image- and label dissimilarity functions, transformation models [@ashburner2007fast; @vercauteren2009diffeomorphic; @hill2001medical], deformation regularisation [@rueckert1999nonrigid] and different neural network architectures [@hu2018weakly; @he2016deep; @simonyan2014very]. Details of the implemented methods are described in the documentation. The provided dataset loaders adopt staged random sampling strategy to ensure unbiased learning from groups, images and labels [@hu2018weakly; @yang2020longitudinal]. These algorithmic components together with the flexible dataset loaders are building blocks of many other registration tasks, such as group-wise registration and morphological template construction [@dalca2019learning; @siebert2020deep; @luo2020mvmm].

# DeepReg Demos
In addition to the tutorials and documentation, DeepReg provides a collection of demonstrations, _DeepReg Demos_, using open-accessible data with real-world clinical applications.

## Paired images
Many clinical applications for tracking organ motion and other temporal changes require _intra-subject_ _single-modality_ image registration. Registering lung CT images for the same patient, acquired at expiratory and inspiratory phases [@hering_alessa_2020_3835682], is such an example of both unsupervised (without labels) and combined supervision (trained with additional label dissimilarity based on anatomical segmentation). Furthermore, registering prostate MR, acquired before surgery, and intra-operative ultrasound images is an example of weakly-supervised learning for multimodal image registration [@hu2018weakly]. Another DeepReg Demo illustrates MR-to-ultrasound image registration is to track tissue deformation and brain tumour resection during neurosurgery [@xiao2017resect].

## Unpaired images
Unpaired images are found in applications such as _single-modality_ _inter-subject_ registration. One demo registers different brain MR images from different subjects [@simpson2019large], fundamental to population studies. Two other applications align unpaired inter-subject CT images for lung [@hering_alessa_2020_3835682] and abdominal organs [@adrian_dalca_2020_3715652]. Additionally, the support for cross-validation in DeepReg has been included in a demo, which registers 3D ultrasound images from different prostate cancer patients.

## Grouped images
Unpaired images may also be grouped in applications such as _single-modality_ _intra-subject_ registration. In this case, each subject has multiple images acquired, for instance, at two or more time points. For demonstration, multi-sequence cardiac MR images, acquired from myocardial infarction patients [@zhuang2020cardiac], are registered, where multiple images within each subject are considered as grouped images. Prostate longitudinal MR registration is proposed to track the cancer progression during active surveillance programme [@yang2020longitudinal]. Using segmentation from this application, another demo application illustrates aligning intra-patient prostate gland masks - also an example of feature-based registration based on deep learning.

# Conclusion
DeepReg provides a collection of deep learning algorithms and dataset loaders to train image registration networks, which provides a reference of basic functionalities. In its permissible open-source format, DeepReg not only provides a tool for scientific research and higher education, but also welcomes contributions from wider communities.

# Acknowledgements

This work is supported by the Wellcome/EPSRC Centre for Interventional and Surgical Sciences (203145Z/16/Z). Support was also from the Engineering and Physical Sciences Research Council (EPSRC) (EP/M020533/1, NS/A000049/1), National Institute for Health Research University College London Hospitals Biomedical Research Centre and Wellcome Trust (203148/Z/16/Z). TV is supported by a Medtronic / Royal Academy of Engineering Research Chair (RCSRF1819\7\34). NMB, ZB, RD are also supported by the EPSRC CDT i4health (EP/S021930/1). ZB is supported by the Natural Sciences and Engineering Research Council of Canada Postgraduate Scholarships-Doctoral Program and the University College London Overseas and Graduate Research Scholarships.

# References
<!-- This will be filled in by references in paper.bib -->
