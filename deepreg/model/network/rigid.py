# # written by minzhe
# # to implement the rigid transformation
# import tensorflow as tf

# from deepreg.model import layer, layer_util
# from deepreg.model.network.util import (
#     add_ddf_loss,
#     add_image_loss,
#     add_label_loss,
#     build_backbone,
#     build_inputs,
# )

# # TO BE DELETED
# moving_image_shape = [2, 512, 512, 512]
# moving_image1 = tf.random.uniform(
#     moving_image_shape,
#     minval=0,
#     maxval=1,
#     dtype=tf.dtypes.float32,
#     seed=None,
#     name=None,
# )
# fixed_image_shape = [2, 256, 256, 256]
# fixed_image1 = tf.random.uniform(
#     fixed_image_shape, minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
# )
# print("moving_image1", moving_image1)
# # first, the moving image is linearly resized to the
# # size as the fixed image
# # need to be squeezed later for warping
# moving_image = tf.expand_dims(
#     moving_image1, axis=4
# )  # (batch, m_dim1, m_dim2, m_dim3, 1)
# print("moving_image", moving_image)  # to be deleted
# fixed_image = tf.expand_dims(fixed_image1, axis=4)  # (batch, f_dim1, f_dim2, f_dim3, 1)

# fixed_image_size = [256, 256, 256]
# # adjust moving image
# moving_image = layer_util.resize3d(
#     image=moving_image, size=fixed_image_size
# )  # (batch, f_dim1, f_dim2, f_dim3, 1)
# print(moving_image)
