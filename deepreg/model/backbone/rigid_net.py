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
#     theta_array = tf.make_ndarray(tf.make_tensor_proto(theta))
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
# # affine_test = rigid2affine( theta, translation )


# # change the rotation and translation into the format of
# # the original affine transformation matrix
# # def rigid2affine( theta, translation )-> tf.Tensor:

# # change the angle to the rotation
# rotation = euler2rot(theta)

# # change tensor to numpy array
# rotation_array = tf.make_ndarray(tf.make_tensor_proto(rotation))
# translation_array = tf.make_ndarray(tf.make_tensor_proto(translation))

# # get the affine transformation matrix in numpy
# affine = np.zeros((4, 3))
# # affine[0:2,0:2] = rotation_array
# # affine[3,0:2]   = translation_array

# # convert the affine into tensor
# affine = tf.convert_to_tensor(affine, dtype=tf.float32)
# # return affine


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
