# Image Registration with Deep Learning

A great scientific tutorial on deep learning for registration can be found at the
[learn2reg tutorial](https://learn2reg.github.io/), held in conjunction with
MICCAI 2019. This tutorial provides a practical overview for a number of algorithms
supported by DeepReg.

## Registration

Image registration is the process of mapping the coordinate system of one image into
another image. A registration method takes a pair of images as input, denoted as moving
and fixed images. In this tutorial, we register the moving image into the fixed image,
i.e. we map the coordinates of the moving image into the fixed image.

<!---
@Yunguan
We could provide some clinical applications of registration.

Personally, this page is for people do not understand the registration,
or people who do not know our work very well.
Sampling options are too advanced and maybe not related.

They can be simply a random pair of images from all the training images
available. They may however require more advanced sampling. For instance, when multiple
subjects each having multiple available images, please see more sampling options in
[Training data sampling options](tutorial_sampling.md).
-->

## Network

- **Predict a dense displacement field**

  With deep learning, given a pair of moving and fixed images, the registration network
  outputs a dense displacement field (DDF) of the same shape of moving image. Each value
  can be considered as the placement of the corresponding pixel / voxel of the moving
  image. Therefore, the DDF defines a mapping from the moving image's coordinates to the
  fixed image.

  In this tutorial, we mainly focus on DDF-based methods.

- **Predict a dense velocity field**

  Another option is topredict a dense velocity field (DVF), such that a diffeomorphic
  DDF can be numerically integrated. Read
  ["A fast diffeomorphic image registration algorithm"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.474.1033&rep=rep1&type=pdf)
  for more details.

- **Predict an affine transformation**

  A more constrained option is to predict an affine transformation, parameterised the
  affine transformation matrix of 12 degrees of freedom. The DDF can then be computed to
  resample the moving images in fixed image space.

- **Predict a region of interest**

  Instead of outputting the transformation between coordinates, given moving image,
  fixed image, and a region of interest (ROI) in the moving image, the network can
  predict the ROI in fixed image directly. Interested readers are referred to the MICCAI
  2019 paper:
  [Conditional segmentation in lieu of image registration](https://arxiv.org/abs/1907.00438)

## Loss

A loss function has to be defined to train a deep neural network. There are mainly three
types of losses:

- **Intensity based (image based) loss**

  This type of loss measures the dissimilarity of the fixed image and warped moving
  image, which is adapted from the classical image registration methods. Intensity based
  loss is modality-independent and similar to many other well-studied computer vision
  and medical imaging tasks, such as image segmentation.

  The common loss functions are normalized cross correlation (NCC), sum of squared
  distance (SSD), and normalized mutual information (MI).

- **Feature based (label based) loss**

  This type of loss measures the dissimilarity of the fixed image labels and warped
  moving image labels. The label is often an ROI in the image, like the segmentation of
  an organ in a CT image.

  The common loss function is Dice loss.

- **Deformation loss**

  This type of loss measures the deformation to regularize the transformation.

  For DDF, the common loss functions are bending energy, L1 or L2 norm of the
  displacement gradient.

## Learning

Depending on the availability of the data labels, we can split the registration network
training into the following types:

### Unsupervised

When the data label is unavailable, the training is driven by the unsupervised loss. The
loss function often consists of the intensity based loss and deformation loss. Following
is an illustration of an unsupervised DDF-based registration network.

![Unsupervised DDF-based registration network](asset/registration-ddf-nn-unsupervised.svg ":size=600")

### Weakly-supervised

The training may take an additional pair of corresponding moving and fixed labels,
represented by binary masks, to compute a label dissimilarity (feature based loss) to
drive the registration.

Combined with the regularisation on the predicted displacement field, this forms a
weakly-supervised training. An illustration of an weakly-supervised DDF-based
registration network is provided below.

When multiple labels are available for each image, the labels can be sampled during
training iteration, such that only one label per image is used in each iteration of the
data set (epoch). Read [data sampling API](tutorial_sampling.md) for more details.

![Weakly-supervised DDF-based registration network](asset/registration-ddf-nn-weakly-supervised.svg ":size=600")

### Combined

When the data label is available, combining intensity based, feature based, and
deformation based losses together has shown superior registration accuracy, compared to
unsupervised and weakly supervised methods. Following is an illustration of a combined
DDF-based registration network.

![Combined DDF-based registration network](asset/registration-ddf-nn-combined.svg ":size=600")
