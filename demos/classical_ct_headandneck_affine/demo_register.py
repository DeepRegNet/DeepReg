"""
A DeepReg Demo for classical affine iterative pairwise registration algorithms
"""
import os

import h5py
import matplotlib.pyplot as plt
import tensorflow as tf

import deepreg.model.layer_util as util
import deepreg.model.loss.image as image_loss

current_path = os.getcwd()
PROJECT_DIR = r"demos/classical_ct_headandneck_affine"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
FILE_PATH = os.path.join(DATA_PATH, "demo.h5")

## registration parameters
image_loss_name = "ssd"
learning_rate = 0.01
total_iter = int(1000)


## load image
if not os.path.exists(DATA_PATH):
    raise ("Download the data using demo_data.py script")
if not os.path.exists(FILE_PATH):
    raise ("Download the data using demo_data.py script")

fid = h5py.File(FILE_PATH, "r")
fixed_image = tf.cast(tf.expand_dims(fid["image"], axis=0), dtype=tf.float32)
fixed_image = (fixed_image - tf.reduce_min(fixed_image)) / (
    tf.reduce_max(fixed_image) - tf.reduce_min(fixed_image)
)  # normalisation to [0,1]

# generate a radomly-affine-transformed moving image
fixed_image_size = fixed_image.shape
transform_random = util.random_transform_generator(batch_size=1, scale=0.2)
grid_ref = util.get_reference_grid(grid_size=fixed_image_size[1:4])
grid_random = util.warp_grid(grid_ref, transform_random)
moving_image = util.resample(vol=fixed_image, loc=grid_random)
# warp the labels to get ground-truth using the same random affine, for validation
fixed_labels = tf.cast(tf.expand_dims(fid["label"], axis=0), dtype=tf.float32)
moving_labels = tf.stack(
    [
        util.resample(vol=fixed_labels[..., idx], loc=grid_random)
        for idx in range(fixed_labels.shape[4])
    ],
    axis=4,
)


## optimisation
@tf.function
def train_step(grid, weights, optimizer, mov, fix):
    """
    Train step function for backprop using gradient tape

    :param grid: reference grid return from util.get_reference_grid
    :param weights: trainable affine parameters [1, 4, 3]
    :param optimizer: tf.optimizers
    :param mov: moving image [1, m_dim1, m_dim2, m_dim3]
    :param fix: fixed image [1, f_dim1, f_dim2, f_dim3]
    :return loss: image dissimilarity to minimise
    """
    with tf.GradientTape() as tape:
        pred = util.resample(vol=mov, loc=util.warp_grid(grid, weights))
        loss = image_loss.dissimilarity_fn(
            y_true=fix, y_pred=pred, name=image_loss_name
        )
    gradients = tape.gradient(loss, [weights])
    optimizer.apply_gradients(zip(gradients, [weights]))
    return loss


# affine transformation as trainable weights
var_affine = tf.Variable(
    initial_value=[
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    ],
    trainable=True,
)
optimiser = tf.optimizers.Adam(learning_rate)
for step in range(total_iter):
    loss_opt = train_step(grid_ref, var_affine, optimiser, moving_image, fixed_image)
    if (step % 50) == 0:  # print info
        tf.print("Step", step, image_loss_name, loss_opt)


## warp the moving image using the optimised affine transformation
grid_opt = util.warp_grid(grid_ref, var_affine)
warped_moving_image = util.resample(vol=moving_image, loc=grid_opt)
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


## warp the moving labels using the optimised affine transformation
warped_moving_labels = tf.stack(
    [
        util.resample(vol=moving_labels[..., idx], loc=grid_opt)
        for idx in range(fixed_labels.shape[4])
    ],
    axis=4,
)

# display
for idx_label in range(fixed_labels.shape[4]):
    plt.figure()
    for idx in range(len(idx_slices)):
        axs = plt.subplot(nIdx, 3, 3 * idx + 1)
        axs.imshow(moving_labels[0, ..., idx_slices[idx], idx_label], cmap="gray")
        axs.axis("off")
        axs = plt.subplot(nIdx, 3, 3 * idx + 2)
        axs.imshow(fixed_labels[0, ..., idx_slices[idx], idx_label], cmap="gray")
        axs.axis("off")
        axs = plt.subplot(nIdx, 3, 3 * idx + 3)
        axs.imshow(
            warped_moving_labels[0, ..., idx_slices[idx], idx_label], cmap="gray"
        )
        axs.axis("off")
    plt.ion()
    plt.show()

plt.pause(0.001)
input("Press [enter] to continue.")

os.chdir(current_path)
