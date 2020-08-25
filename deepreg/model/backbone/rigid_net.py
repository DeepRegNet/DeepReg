# # to be deleted
# # six parameters are utilized


# import math
# import os

# import numpy as np
# import tensorflow as tf

# import deepreg.model.backbone.global_net as g
# from deepreg.dataset.loader.h5_loader import H5FileLoader

# # from deepreg.model import layer, layer_util
# from deepreg.model import layer_util

# # from deepreg.predict import predict
# from deepreg.train import train

# #  coding=utf-8

# """
# Module to build GlobalNet based on:

# Y. Hu et al.,
# "Label-driven weakly-supervised learning for multimodal
# deformable image registration,"
# (ISBI 2018), pp. 1070-1074.
# https://ieeexplore.ieee.org/abstract/document/8363756?casa_token=FhpScE4qdoAAAAAA:dJqOru2PqjQCYm-n81fg7lVL5fC7bt6zQHiU6j_EdfIj7Ihm5B9nd7w5Eh0RqPFWLxahwQJ2Xw
# """
# import tensorflow as tf

# from deepreg.model import layer, layer_util
# import deepreg.model.backbone.global_net as g

# out = 3
# im_size = [1, 2, 3]
# #  Initialising GlobalNet instance
# global_test = g.GlobalNet(
#         image_size=im_size,
#         out_channels=out,
#         num_channel_initial=3,
#         extract_levels=[1, 2, 3],
#         out_kernel_initializer="softmax",
#         out_activation="softmax",
#     )
# # Pass an input of all zeros
# inputs = tf.constant(
#         np.zeros((5, im_size[0], im_size[1], im_size[2], out), dtype=np.float32)
# )
# #  Get outputs by calling
# output = global_test.call(inputs)

# class RigidNet(tf.keras.Model):
#     """
#     Write the RigidNet for the rigid alignmnet of two images
#     based on the global_net
#     """

#     def __init__(
#         self,
#         image_size,
#         out_channels,
#         num_channel_initial,
#         extract_levels,
#         out_kernel_initializer,
#         out_activation,
#         **kwargs,
#     ):
#         """
#         Image is encoded gradually, i from level 0 to E.
#         Then, a densely-connected layer outputs an affine
#         transformation.
#         :param out_channels: int, number of channels for the output
#         :param num_channel_initial: int, number of initial channels
#         :param extract_levels: list, which levels from net to extract
#         :param out_activation: str, activation at last layer
#         :param out_kernel_initializer: str, which kernel to use as initialiser
#         :param kwargs:
#         """
#         super(RigidNet, self).__init__(**kwargs)

#         # save parameters
#         self._extract_levels = extract_levels
#         self._extract_max_level = max(self._extract_levels)  # E
#         self.reference_grid = layer_util.get_reference_grid(image_size)
#         self.transform_initial = tf.constant_initializer(value=[0,0,0,0,0,0])
#         # init layer variables
#         num_channels = [
#             num_channel_initial * (2 ** level)
#             for level in range(self._extract_max_level + 1)
#         ]  # level 0 to E
#         self._downsample_blocks = [
#             layer.DownSampleResnetBlock(
#                 filters=num_channels[i], kernel_size=7 if i == 0 else 3
#             )
#             for i in range(self._extract_max_level)
#         ]  # level 0 to E-1
#         self._conv3d_block = layer.Conv3dBlock(filters=num_channels[-1])  # level E
#         self._dense_layer = layer.Dense(
#             units=6, bias_initializer=self.transform_initial
#         )

#     def call(self, inputs, training=None, mask=None):
#         """
#         Build GlobalNet graph based on built layers.
#         :param inputs: image batch, shape = [batch, f_dim1, f_dim2, f_dim3, ch]
#         :param training:
#         :param mask:
#         :return:
#         """
#         # down sample from level 0 to E
#         h_in = inputs
#         for level in range(self._extract_max_level):  # level 0 to E - 1
#             h_in, _ = self._downsample_blocks[level](inputs=h_in, training=training)
#         h_out = self._conv3d_block(
#             inputs=h_in, training=training
#         )  # level E of encoding

#         # predict affine parameters theta of shape = [batch, 4, 3]
#         self.theta = self._dense_layer(h_out)
#         self.transformation = rigid2affine( self.theta[0:3], self.theta[3:6] )
#         self.theta = tf.reshape(self.theta, shape=(-1, 4, 3))
#         # warp the reference grid with affine parameters to output a ddf
#         grid_warped = layer_util.warp_grid(self.reference_grid, self.theta)
#         output = grid_warped - self.reference_grid
#         return output

# # from Euler angles to the rotation matrix
# # here is the function that transform the euler angles
# # to a rotation matrix
# def euler2rot(theta) -> tf.Tensor:
#     """
#     convert a euler angles to a rotation matrix
#     :param theta: is a three-dimensional vector that can be euler angles
#     :return: shape (3,3)
#     """
#     theta = tf.ones((3))
#     #theta_array = tf.make_ndarray(tf.make_tensor_proto(theta))
#     theta_array = np.asarray(theta) # use this one to conver the tensor to
#     rotation_shape = (3, 3)
#     rotation = np.zeros(rotation_shape)
#     theta0 = theta_array[0]
#     theta1 = theta_array[1]
#     theta2 = theta_array[2]
#     rotation[0, 0] = math.cos(theta1) * math.cos(theta2)
#     rotation[0, 1] = math.sin(theta0) * math.sin(theta1) * math.cos(theta2) - math.cos(
#         theta0
#     ) * math.sin(theta2)
#     rotation[0, 2] = math.cos(theta0) * math.sin(theta1) * math.cos(theta2) + math.sin(
#         theta0
#     ) * math.sin(theta2)
#     rotation[1, 0] = math.cos(theta1) * math.sin(theta2)
#     rotation[1, 1] = math.sin(theta0) * math.sin(theta1) * math.cos(theta2) + math.cos(
#         theta0
#     ) * math.sin(theta2)
#     rotation[1, 2] = math.cos(theta0) * math.sin(theta1) * math.cos(theta2) - math.sin(
#         theta0
#     ) * math.sin(theta2)
#     rotation[2, 0] = -math.sin(theta1)
#     rotation[2, 1] = math.sin(theta0) * math.cos(theta1)
#     rotation[2, 2] = math.cos(theta0) * math.cos(theta1)

#     rotation = tf.convert_to_tensor(rotation, dtype=tf.float32)
#     return rotation


# # test rigid2affine
# theta = tf.zeros((3))
# translation = tf.ones((3))
# affine_test = rigid2affine( theta, translation )


# # change the rotation and translation into the format of
# # the original affine transformation matrix
# def rigid2affine(theta,translation )-> tf.Tensor:
#     # change the angle to the rotation
#     rotation = euler2rot(theta)
#     # change tensor to numpy array
#     rotation_array = tf.make_ndarray(tf.make_tensor_proto(rotation))
#     translation_array = tf.make_ndarray(tf.make_tensor_proto(translation))
#     # get the affine transformation matrix in numpy
#     affine = np.zeros((4, 3))
#     affine[0:3,:] = rotation_array[:,:] # from 0 to 2(be reached)
#     affine[3,:]   = translation_array
#     # convert the affine into tensor
#     affine = tf.convert_to_tensor(affine, dtype=tf.float32)
#     return affine

# def rigid2affinebatch(rigid_batch)->tf.Tensor:
#     """
#     the input:  rigid_batch's size is [batch, 6], corresponds to theta in the global_net.py
#     the output: is the affine in batch [batch, 4, 3]
#     """
#     # convert the [batch, 6] to [batch, 4, 3]

#     # First step, access the first three elements in theta
#     theta = tf.constant(np.zeros((5, 6), dtype=np.float32))
#     angle = theta[:, 0:3]
#     # Second step, change the angle into rotation
#     # for x in fruits:


#     # return affine


# grid_size = [4, 4, 4]
# reference_grid = layer_util.get_reference_grid(grid_size)
# print(reference_grid)  #
# theta = tf.random.uniform(shape=[1, 4, 3])  # the shape is 4*3

# grid_warped = layer_util.warp_grid(reference_grid, theta)


# dir_path = "./data/test/h5/paired/test"
# name = "fixed_images"

# loader = H5FileLoader(dir_path=dir_path, name=name, grouped=False)
# index = 0
# array = loader.get_data(index)
# got = [np.shape(array), [np.amax(array), np.amin(array), np.mean(array), np.std(array)]]
# expected = [(44, 59, 41), [255.0, 0.0, 68.359276, 65.84009]]
# print(loader.get_num_images())  # the number is one
# print(loader.h5_file.filename)  # ./data/test/h5/paired/test/fixed_images.h5
# print(loader.data_keys)  # ['case000025.nii.gz']
# print(loader.get_data_ids())  # ['case000025.nii.gz']


# gpu = ""
# gpu_allow_growth = False

# train(
#     gpu=gpu,
#     config_path="deepreg/config/unpaired_labeled_ddf.yaml",
#     gpu_allow_growth=gpu_allow_growth,
#     ckpt_path="",
#     log_dir="test_train",
# )


# print(
#     os.path
# )  # <module 'posixpath' from '/home/linux-min/miniconda3/envs/deepreg/lib/python3.7/posixpath.py'>

# # test the call function of GlobalNet
# out = 3
# im_size = [1, 2, 3]
# #  Initialising GlobalNet instance
# global_test = g.GlobalNet(
#     image_size=im_size,
#     out_channels=out,
#     num_channel_initial=3,
#     extract_levels=[1, 2, 3],
#     out_kernel_initializer="softmax",
#     out_activation="softmax",
# )
# # Pass an input of all zeros
# inputs = tf.constant(
#     np.zeros((5, im_size[0], im_size[1], im_size[2], out), dtype=np.float32)
# )
# #  Get outputs by calling
# output = global_test.call(inputs)
# print(output)  # shape=(5, 1, 2, 3, 3), dtype=float32)
# print(output)

# #  Initialising GlobalNet instance
# global_test = g.GlobalNet(
#     image_size=[1, 2, 3],
#     out_channels=3,
#     num_channel_initial=3,
#     extract_levels=[1, 2, 3],
#     out_kernel_initializer="softmax",
#     out_activation="softmax",
# )
# print(global_test._extract_levels)  # ListWrapper([1, 2, 3])
# print(
#     type(global_test._extract_levels)
# )  # <class 'tensorflow.python.training.tracking.data_structures.ListWrapper'>
# print(global_test.reference_grid)
# # tf.Tensor(
# # [[[[0. 0. 0.]
# #    [0. 0. 1.]
# #    [0. 0. 2.]]
# #   [[0. 1. 0.]
# #    [0. 1. 1.]
# #    [0. 1. 2.]]]], shape=(1, 2, 3, 3), dtype=float32)


# print(global_test.transform_initial)
# # <tensorflow.python.ops.init_ops_v2.Constant object at 0x7feaa979c490>
# print(global_test.transform_initial.value)  # print(global_test.transform_initial.value)

# print(type(global_test.transform_initial.value))  # <class 'list'>
# # get one value from the list
# print(global_test.transform_initial.value[0])  # the first element
# # the result is 1.0
# print(
#     global_test.transform_initial.value[-1]
# )  # the last element of the list, the result is 0.0

# print(
#     global_test._downsample_blocks
# )  # ListWrapper([<deepreg.model.layer.DownSampleResnetBlock object at 0x7f901d55cc10>, <deepreg.model.layer.DownSampleResnetBlock object at 0x7f901842ba90>, <deepreg.model.layer.DownSampleResnetBlock object at 0x7f9018451f90>])

# print("global_test._conv3d_block", global_test._conv3d_block)  # 0x7f128c01d210>

# print(
#     "global_test._dense_layer", global_test._dense_layer
# )  # <deepreg.model.layer.Dense object at 0x7f5fd78d8710>

# print(global_test)  # <deepreg.model.backbone.global_net.GlobalNet


# # test the call function
