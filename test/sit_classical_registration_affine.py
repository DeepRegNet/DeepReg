"""
Classical affine iterative pairwise registration algorithms as integration tests
"""
import nibabel
import tensorflow as tf 
import deepreg.model.layer as layer
import deepreg.model.loss.image as image_loss
import deepreg.model.layer_util as layer_util
import matplotlib.pyplot as plt


## registration parameters
image_loss_name = 'ssd'
learning_rate = 0.1
total_iter = int(3000)


## load image
load_image = lambda fn:tf.cast(tf.expand_dims(nibabel.load(fn).dataobj, axis=0), dtype=tf.float32)
moving_image = load_image('./data/mr_us/unpaired/train/images/case000000.nii.gz')
# fixed_image = load_image('./data/mr_us/unpaired/train/images/case000001.nii.gz')

# random affine-transformed fixed image
fixed_image_size = moving_image.shape
random_transform = layer_util.random_transform_generator(batch_size=1, scale=0.2)
grid_ref = layer_util.get_reference_grid(grid_size=fixed_image_size[1:4])
fixed_image = layer_util.resample(vol=moving_image, loc=layer_util.warp_grid(grid_ref, random_transform))


## optimisation
@tf.function
def train_step(grid, weights, optimizer, mov, fix):
    """
    Train step function for backprop using gradient tape    

    :param grid: reference grid return from layer_util.get_reference_grid
    :param weights: trainable affine parameters [1, 4, 3]
    :param optimizer: tf.optimizers
    :param mov: moving image [1, m_dim1, m_dim2, m_dim3]
    :param fix: fixed image [1, f_dim1, f_dim2, f_dim3]
    :return loss: image dissimilarity to minimise
    """
    with tf.GradientTape() as tape:
        pred = layer_util.resample(vol=mov, loc=layer_util.warp_grid(grid, weights))
        loss = image_loss.similarity_fn(y_true=fix, y_pred=pred, name=image_loss_name)
    gradients = tape.gradient(loss, [weights])
    optimizer.apply_gradients(zip(gradients, [weights]))
    return loss

# affine transformation as trainable weights
var_affine = tf.Variable(initial_value=[[1,0,0],[0,1,0],[0,0,1],[0,0,0]], trainable=True)
optimiser = tf.optimizers.Adam(learning_rate)
for step in range(total_iter):
    loss_opt = train_step(grid_ref, var_affine, optimiser, moving_image, fixed_image)
    if (step % 50) == 0:  # print info
        tf.print('Step',step, image_loss_name,loss_opt)


## predict
pred_fixed_image = layer_util.resample(vol=moving_image, loc=layer_util.warp_grid(grid_ref, var_affine))

# display
idx_slices = [int(5+x*5) for x in range(int(fixed_image_size[3]/5)-1)]
nIdx = len(idx_slices)
plt.figure()
for idx in range(len(idx_slices)):
    axs = plt.subplot(nIdx, 3, 3*idx+1)
    axs.imshow(moving_image[0,...,idx_slices[idx]], cmap='gray')
    axs.axis('off')
    axs = plt.subplot(nIdx, 3, 3*idx+2)
    axs.imshow(fixed_image[0,...,idx_slices[idx]], cmap='gray')
    axs.axis('off')
    axs = plt.subplot(nIdx, 3, 3*idx+3)
    axs.imshow(pred_fixed_image[0,...,idx_slices[idx]], cmap='gray')
    axs.axis('off')
plt.show()