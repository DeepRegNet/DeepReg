# Get started with image registration using deep learning

A great scientific tutorial on deep learning for registration can be found at the [learn2reg tutorial](https://learn2reg.github.io/), held in conjunction with MICCAI 2019. This tutorial provides a practical overview for a number of algorithms supported by DeepReg.

## Decide what the network input images are

A registration network takes a pair of moving and fixed images as the input of the deep nerual network. what types of the images are the input depend on the clinical application. They can be simply a random pair of images from all the training images available. They may however require more advanced sampling. For instance, when multiple subjects each having multiple available images, please see more sampling options in [Training data sampling options](./sampling.md).

## Network outputs

The registration network outputs a dense displacement field (DDF). The DDF can be constrained during training by a deformation regularisation term. The deformation regularisation include, L1- or L2 norm of the displacement gradient and bending energy.

Option to predict a dense velocity field is also available, such that a diffeomorphic DDF can be numerically integrated.

A more constrained option is to predict an affine transformation, parameterised by 12 parameters of the affine transformation matrix. The DDF can then be computed to resample the moving images in fixed image space.

## Loss function

In addition to the deformation regularisation, the loss function to train a registration network depends on the adopted algorithms.

### Unsupervised learning

For unsupervised learning, the training is driven by the unsupervised loss. The loss functions are often consisted of a deformation regularisation term on the predicted displacement field and an image dissimilarity measure between the fixed and warped moving images, which are adapted from the claissical image registration methods. The image dissimilarity measures include sum-of-square difference in intensity (SSD), normalised cross correlation (NCC and normalised mutual information (MI).

<img src="./media/deepreg-tutorial-unsupervised.svg" alt="" title="unsupervised" width="300" />

### Weakly-supervised learning

The training may take an additional pair of corresponding moving and fixed labels, represented by binary masks, to compute a label dissimilarity to drive the registration.

In addition to the regularisation on the predicted displacement field, the training is driven by minimising the dissimilarity between the fixed labels and warped moving labels, one that is modality-independent and similar to many other well-studied computer vision and medical imaging tasks, such as image segmentation.

When multiple labels are available for each image, the labels can be sampled during training iteration, such that on one label with one image is used in each iteration, that is a pair of moving and fixed images and a pair of moving and fixed labels being loaded into training. See other sampling options in [Training data sampling options](./sampling.md).

<img src="./media/deepreg-tutorial-weakly.svg" alt="" title="weakly" width="300" />

### Unsupervised learning with weak supervision

Combining the unsupervised loss and the weak supervision has shown superior registration accuracy, compared with that using unsupervised loss alone.

<img src="./media/deepreg-tutorial-combined.svg" alt="" title="combined" width="300" />

### Conditional segmentation

Depite the name, this formulation to predcit the corresponding regions of interest is considered a image registration, rather than a image segmentation algorithm. Interested readers are refered to the MICCAI 2019 paper:
'Hu, Y., Gibson, E., Barratt, D.C., Emberton, M., Noble, J.A. and Vercauteren, T., 2019, October. Conditional segmentation in lieu of image registration. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 401-409). Springer, Cham.'
[paper link](https://arxiv.org/abs/1907.00438)

## Training a registration network

Following these simplpe steps:

- [Setup](./setup.md);

- Data. For using predefined data loaders, see the [tutorial](./predefined_loader.md). Otherise, a [cutomised data loader](./add_loader.md) is required;

- [Configure](./configuration.md) - see an overview of other configuration options;

- Train by calling 'train.py'.

- Predict by calling 'predict.py'.

- Demos are available for example applications [link to demo overview](./demo.md)

- Experiments often require random-split or cross-validation. This is likely to be supported in a future release. There are some [workaround](/experiment.md) using the exisiting folder-based data loaders for these.

- Other advanced uses include [add a new loss](./add_loss.md) and [add a new network](./add_network.md)
