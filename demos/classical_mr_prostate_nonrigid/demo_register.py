"""
A DeepReg Demo for classical nonrigid iterative pairwise registration algorithms
"""
import os

import h5py
import matplotlib.pyplot as plt
import tensorflow as tf

import deepreg.model.layer as layer
import deepreg.model.loss.deform as deform_loss
import deepreg.model.loss.image as image_loss

current_path = os.getcwd()
PROJECT_DIR = r"demos/classical_mr_prostate_nonrigid"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
FILE_PATH = os.path.join(DATA_PATH, "demo2.h5")


## registration parameters
image_loss_name = "lncc"
deform_loss_name = "bending"
weight_deform_loss = 1e3
learning_rate = 0.1
total_iter = int(3000)


## load images
if not os.path.exists(DATA_PATH):
    raise ("Download the data using demo_data.py script")
if not os.path.exists(FILE_PATH):
    raise ("Download the data using demo_data.py script")

fid = h5py.File(FILE_PATH, "r")
moving_image = tf.cast(tf.expand_dims(fid["image0"], axis=0), dtype=tf.float32)
fixed_image = tf.cast(tf.expand_dims(fid["image1"], axis=0), dtype=tf.float32)


## optimisation
@tf.function
def train_step(warper, weights, optimizer, mov, fix):
    """
    Train step function for backprop using gradient tape

    :param warper: warping function returned from layer.Warping
    :param weights: trainable ddf [1, f_dim1, f_dim2, f_dim3, 3]
    :param optimizer: tf.optimizers
    :param mov: moving image [1, m_dim1, m_dim2, m_dim3]
    :param fix: fixed image [1, f_dim1, f_dim2, f_dim3]
    :return:
        loss: overall loss to optimise
        loss_image: image dissimilarity
        loss_deform: deformation regularisation
    """
    with tf.GradientTape() as tape:
        pred = warper(inputs=[weights, mov])
        loss_image = image_loss.dissimilarity_fn(
            y_true=fix, y_pred=pred, name=image_loss_name
        )
        loss_deform = deform_loss.local_displacement_energy(weights, deform_loss_name)
        loss = loss_image + weight_deform_loss * loss_deform
    gradients = tape.gradient(loss, [weights])
    optimizer.apply_gradients(zip(gradients, [weights]))
    return loss, loss_image, loss_deform


# ddf as trainable weights
fixed_image_size = fixed_image.shape
initialiser = tf.random_normal_initializer(mean=0, stddev=1e-3)
var_ddf = tf.Variable(initialiser(fixed_image_size + [3]), name="ddf", trainable=True)

warping = layer.Warping(fixed_image_size=fixed_image_size[1:4])
optimiser = tf.optimizers.Adam(learning_rate)
for step in range(total_iter):
    loss_opt, loss_image_opt, loss_deform_opt = train_step(
        warping, var_ddf, optimiser, moving_image, fixed_image
    )
    if (step % 50) == 0:  # print info
        tf.print(
            "Step",
            step,
            "loss",
            loss_opt,
            image_loss_name,
            loss_image_opt,
            deform_loss_name,
            loss_deform_opt,
        )


## warp the moving image using the optimised ddf
warped_moving_image = warping(inputs=[var_ddf, moving_image])
# display
idx_slices = [int(5 + x * 5) for x in range(int(fixed_image_size[3] / 5) - 1)]
nIdx = len(idx_slices)
plt.figure()
for idx in range(len(idx_slices)):
    axs = plt.subplot(nIdx, 3, 3 * idx + 1)
    axs.imshow(moving_image[0, ..., idx_slices[idx]], cmap="gray")
    axs.axis("off")
    axs = plt.subplot(nIdx, 3, 3 * idx + 2)
    axs.imshow(fixed_image[0, ..., idx_slices[idx]], cmap="gray")
    axs.axis("off")
    axs = plt.subplot(nIdx, 3, 3 * idx + 3)
    axs.imshow(warped_moving_image[0, ..., idx_slices[idx]], cmap="gray")
    axs.axis("off")
plt.ion()
plt.show()


## warp the moving label using the optimised affine transformation
moving_label = tf.cast(tf.expand_dims(fid["label0"], axis=0), dtype=tf.float32)
fixed_label = tf.cast(tf.expand_dims(fid["label1"], axis=0), dtype=tf.float32)
warped_moving_label = warping(inputs=[var_ddf, moving_label])
# display
plt.figure()
for idx in range(len(idx_slices)):
    axs = plt.subplot(nIdx, 3, 3 * idx + 1)
    axs.imshow(moving_label[0, ..., idx_slices[idx]], cmap="gray")
    axs.axis("off")
    axs = plt.subplot(nIdx, 3, 3 * idx + 2)
    axs.imshow(fixed_label[0, ..., idx_slices[idx]], cmap="gray")
    axs.axis("off")
    axs = plt.subplot(nIdx, 3, 3 * idx + 3)
    axs.imshow(warped_moving_label[0, ..., idx_slices[idx]], cmap="gray")
    axs.axis("off")
plt.ion()
plt.show()

plt.pause(0.001)
input("Press [enter] to continue.")

os.chdir(current_path)
