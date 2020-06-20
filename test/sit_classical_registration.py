
import nibabel
import numpy as np
import tensorflow as tf 
import deepreg.model.layer_util as layer_util
import deepreg.model.layer as layer
import deepreg.model.loss.image as image_loss
import deepreg.model.loss.deform as deform_loss

load_image = lambda fn:tf.cast(tf.expand_dims(nibabel.load(fn).dataobj, axis=0), dtype=tf.float32)
moving_image = load_image('./data/mr_us/unpaired/train/images/case000000.nii.gz')
fixed_image = load_image('./data/mr_us/unpaired/train/images/case000001.nii.gz')

image_loss_name = 'ssd'
deform_loss_name = 'bending'
weight_deform_loss = 100

fixed_image_size = fixed_image.shape
warping = layer.Warping(fixed_image_size=fixed_image_size[1:4])

initialiser = tf.random_normal_initializer(mean=0, stddev=1e-3)
var_ddf = tf.Variable(
    initial_value = initialiser(shape=fixed_image_size+[3]),
    name = 'ddf',
    trainable=True
)


@tf.function
def train_step(wapper, weights, optimizer, mov, fix):
    with tf.GradientTape() as tape:
        pred = wapper(inputs=[weights, mov])
        loss_image = image_loss.similarity_fn(y_true=fix, y_pred=pred, name=image_loss_name)
        loss_deform = deform_loss.local_displacement_energy(weights, deform_loss_name)
        loss = loss_image + weight_deform_loss*loss_deform

    gradients = tape.gradient(loss, [weights])
    optimizer.apply_gradients(zip(gradients, [weights]))
    return loss, loss_image, loss_deform

learning_rate = 0.1
total_iter = int(1000)
optimizer = tf.optimizers.Adam(learning_rate)

# train
for step in range(total_iter):
    loss_train, loss_image_train, loss_deform_train = train_step(warping, var_ddf, optimizer, moving_image, fixed_image)

    if (step % 10) == 0:
        tf.print('Step', step, 
        'loss', loss_train, 
        image_loss_name, loss_image_train, 
        deform_loss_name, loss_deform_train)


# predict
pred_fixed_image = warping(inputs=[var_ddf, moving_image])

import matplotlib.pyplot as plt

print(fixed_image_size)

idx_slices = [20, 25, 30, 35, 40]
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