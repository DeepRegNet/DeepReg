
import nibabel
import tensorflow as tf 
import deepreg.model.layer_util as layer_util
import deepreg.model.layer as layer
import deepreg.model.loss.image as image_loss
import deepreg.model.loss.deform as deform_loss

moving_image = nibabel.load('./data/mr_us/unpaired/train/images/case000000.nii.gz').dataobj
fixed_image = nibabel.load('./data/mr_us/unpaired/train/images/case000001.nii.gz').dataobj

image_loss_name = 'ssd'
deform_loss_name = 'gradient-l2'
weight_deform_loss = 0.1

fixed_image_size = list(fixed_image.shape)
warping = layer.Warping(fixed_image_size=fixed_image_size)

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

    gradients = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(gradients, weights))
    return loss


learning_rate = 1e-5
total_iter = 100
optimizer = tf.optimizers.Adam(learning_rate)
for step in range(total_iter):
    loss_train = train_step(warping, var_ddf, optimizer, moving_image, fixed_image)



pred_fixed_image = warping(inputs=[var_ddf, moving_image])
loss_image = image_loss.similarity_fn(y_true=fixed_image, y_pred=pred_fixed_image, name=image_loss_name)
loss_deform = deform_loss.local_displacement_energy(var_ddf, deform_loss_name)

