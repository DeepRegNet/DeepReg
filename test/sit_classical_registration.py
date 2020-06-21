'''
Classical iterative pairwise registration algorithms as integration tests
'''
import nibabel
import tensorflow as tf 
import deepreg.model.layer as layer
import deepreg.model.loss.image as image_loss
import deepreg.model.loss.deform as deform_loss
import deepreg.model.layer_util as layer_util
import matplotlib.pyplot as plt


## registration parameters
image_loss_name = 'ssd'
deform_loss_name = 'bending'
weight_deform_loss = 1e3
learning_rate = 0.1
total_iter = int(3000)


## load image
load_image = lambda fn:tf.cast(tf.expand_dims(nibabel.load(fn).dataobj, axis=0), dtype=tf.float32)
moving_image = load_image('./data/mr_us/unpaired/train/images/case000000.nii.gz')
# fixed_image = load_image('./data/mr_us/unpaired/train/images/case000001.nii.gz')

# random affine-transformed fixed image
random_transform = layer_util.random_transform_generator(batch_size=1, scale=0.2)
grid_ref = layer_util.get_reference_grid(grid_size=moving_image.shape[1:4])
fixed_image = layer_util.resample(vol=moving_image, loc=layer_util.warp_grid(grid_ref, random_transform))

# ddf as trainable weights
fixed_image_size = fixed_image.shape
initialiser = tf.random_normal_initializer(mean=0, stddev=1e-3)
var_ddf = tf.Variable(initialiser(fixed_image_size+[3]), name='ddf', trainable=True)
warping = layer.Warping(fixed_image_size=fixed_image_size[1:4])


## optimisation
@tf.function
def train_step(warper, weights, optimizer, mov, fix):
    with tf.GradientTape() as tape:
        pred = warper(inputs=[weights, mov])
        loss_image = image_loss.similarity_fn(y_true=fix, y_pred=pred, name=image_loss_name)
        loss_deform = deform_loss.local_displacement_energy(weights, deform_loss_name)
        loss = loss_image + weight_deform_loss*loss_deform
    gradients = tape.gradient(loss, [weights])
    optimizer.apply_gradients(zip(gradients, [weights]))
    return loss, loss_image, loss_deform

optimizer = tf.optimizers.Adam(learning_rate)
for step in range(total_iter):
    loss_opt, loss_image_opt, loss_deform_opt = train_step(warping, var_ddf, optimizer, moving_image, fixed_image)
    if (step % 50) == 0:  # print info
        tf.print('Step',step, 'loss',loss_opt, image_loss_name,loss_image_opt, deform_loss_name,loss_deform_opt)


## predict
pred_fixed_image = warping(inputs=[var_ddf, moving_image])

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