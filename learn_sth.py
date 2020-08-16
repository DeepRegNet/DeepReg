# # written by minzhe
# # August 10th, 2020
# # out_channels = -1
# # raise ValueError(f"out_channels must be int >=1, got {out_channels}")

# # for a_notused, b_notused in enumerate(
# #     range(3, 1, -1)
# #     ):
# #     print('a_notused', a_notused) # 0, 1
# #     print('b_notused', b_notused) # 3, 2
# import tensorflow as tf

# import deepreg.model.loss.deform as deform_loss
# import deepreg.model.loss.image as image_loss
# import deepreg.model.loss.label as label_loss
# from deepreg.model import layer
# from deepreg.model.backbone.global_net import GlobalNet
# from deepreg.model.backbone.local_net import LocalNet
# from deepreg.model.backbone.u_net import UNet

# transform_initial = tf.Variable(
#     [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
# )

# print(
#     transform_initial
# )  # numpy=array([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.], dtype=float32)>

# theta = tf.reshape(transform_initial, shape=(4, 3))
# print(theta)

# extract_max_level = 4
# extract_min_level = 2
# num_channel_initial = 32
# num_channels_automatic = [
#     num_channel_initial * (2 ** level) for level in range(extract_max_level + 1)
# ]
# print(num_channels_automatic)  # [32, 64, 128, 256, 512]

# for level in range(5):
#     print(level)  # the result is 0, 1, 2, 3, 4

# num_channels = [32, 64, 128, 256, 512]  # length is 5
# downsample_blocks = [
#     layer.DownSampleResnetBlock(filters=num_channels[i], kernel_size=7 if i == 0 else 3)
#     for i in range(4)  # five in all
# ]
# print(
#     downsample_blocks
# )  # [<deepreg.model.layer.DownSampleResnetBlock object at 0x7f82ce9247d0>, <deepreg.model.layer.DownSampleResnetBlock object at 0x7f82ce09ae50>, <deepreg.model.layer.DownSampleResnetBlock object at 0x7f82ce043690>, <deepreg.model.layer.DownSampleResnetBlock object at 0x7f82ce060f90>, <deepreg.model.layer.DownSampleResnetBlock object at 0x7f82ce008950>]

# print(num_channels[-1])  # the last element in num_channels, which
# # is 512
# conv3d_block = layer.Conv3dBlock(filters=num_channels[-1])

# print(conv3d_block)  # <deepreg.model.layer.Conv3dBlock object at 0x7f1dc0fae650>

# for level in range(extract_max_level - 1, extract_min_level - 1, -1):  # range(3,1,-1)
#     print(level)  # 3,2 (the 1 can not be reached)

# upsample_blocks = [
#     layer.LocalNetUpSampleResnetBlock(num_channels[level])
#     for level in range(
#         extract_max_level - 1, extract_min_level - 1, -1
#     )  # range(3,1,-1)
# ]
# print(
#     upsample_blocks
# )  # [<deepreg.model.layer.LocalNetUpSampleResnetBlock object at 0x7f32a39479d0>, <deepreg.model.layer.LocalNetUpSampleResnetBlock object at 0x7f32a38de8d0>]
# # D= 2 to 3

# for _ in range(4):
#     print(_)  # 1, 2, 3

# # define the random generated images.
# image_shape_learn = [2, 64, 64, 64, 3]
# random_images_learn = tf.random.uniform(
#     image_shape_learn, minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
# )
# print(random_images_learn)
# print(random_images_learn.shape)  #  (2, 64, 64, 64, 3)

# enum_channelsoded = []
# h_in = random_images_learn
# print(h_in.shape)  # (2, 64, 64, 64, 3)
# for level in range(extract_max_level):  #
#     h_in, h_channel = downsample_blocks[level](inputs=h_in, training=None)
#     print("h_in shape", h_in.shape)  # (2, 32, 32, 32, 32)
#     # (2, 16, 16, 16, 64),(2, 8, 8, 8, 128),(2, 4, 4, 4, 256)
#     print("h_channel shape", h_channel.shape)  # (2, 64, 64, 64, 32)
#     # (2, 32, 32, 32, 64), (2, 16, 16, 16, 128), (2, 8, 8, 8, 256)
#     enum_channelsoded.append(h_channel)
# print("enum_channelsoded", enum_channelsoded)
# print("h_in here", h_in.shape)  # (2, 4, 4, 4, 256)
# h_bottom = conv3d_block(
#     inputs=h_in, training=None
# )  # level E of enum_channelsoding/decoding
# print("h_bottom shape", h_bottom.shape)  # (2, 4, 4, 4, 512)


# # up sample from level E to D
# decoded = [h_bottom]  # level E
# for idx, level in enumerate(
#     range(extract_max_level - 1, extract_min_level - 1, -1)
# ):  # level E-1 to D
#     print("idx", idx)  # 0, 1
#     print("level", level)  # 3,2
#     h_bottom = upsample_blocks[idx](
#         inputs=[h_bottom, enum_channelsoded[level]], training=None
#     )
#     print("h_bottom shape", h_bottom.shape)  # (2, 8, 8, 8, 256)
#     # , (2, 16, 16, 16, 128)
#     decoded.append(h_bottom)
# # print(decoded)
# print("decoded length", len(decoded))  # 3

# # output  ## TO DO LIST!
# # Refer to the layer.py
# out_kernel_initializer = "glorot_uniform"
# extract_levels = [2, 3, 4]
# image_size = image_shape_learn
# out_activation = None
# out_channels = 256
# extract_layers = [
#     # if kernels are not initialized by zeros, with init NN, extract may be too large
#     layer.Conv3dWithResize(
#         output_shape=image_size,
#         filters=out_channels,
#         kernel_initializer=out_kernel_initializer,
#         activation=out_activation,
#     )
#     for _ in extract_levels
# ]
# print(
#     extract_layers
# )  # [<deepreg.model.layer.Conv3dWithResize object at 0x7f792c333fd0>, <deepreg.model.layer.Conv3dWithResize object at 0x7f792c336c90>, <deepreg.model.layer.Conv3dWithResize object at 0x7f792c35d8d0>]

# for idx, level in enumerate(extract_levels):
#     print("idx", idx)  # 0,1,2
#     print("level", level)  # 2,3,4


# output = tf.reduce_mean(
#     tf.stack(
#         [
#             # print('extract_max_level', extract_max_level)
#             # print('level', level)
#             # print(inputs)
#             extract_layers[idx](inputs=decoded[extract_max_level - level])
#             for idx, level in enumerate(extract_levels)
#         ],
#         axis=5,
#     ),
#     axis=5,
# )

# print(output)


# print(loss.shape)
# loss_label = tf.reduce_mean(
#     label_loss.get_dissimilarity_fn(config=loss_config["dissimilarity"]["label"])(
#         y_true=fixed_label, y_pred=pred_fixed_label
#     )
# )

# inputs = tf.keras.Input(shape=(3,))
# print(type(inputs))
# x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
# print(x)
# outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
# print(model)
