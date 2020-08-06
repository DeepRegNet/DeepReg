# # learn the classical non-rigid registration algorithm
# # July 29th, 2020

# import os
# import tkinter

# import h5py
# import matplotlib
# import matplotlib.pyplot as plt
# import tensorflow as tf

# import deepreg.model.layer as layer
# import deepreg.model.loss.deform as deform_loss
# import deepreg.model.loss.image as image_loss

# matplotlib.use("TkAgg")


# MAIN_PATH = os.getcwd()
# print(MAIN_PATH)  #  /home/linux-min/DeepReg/demos/classical_mr_prostate_nonrigid

# PROJECT_DIR = r"demos/classical_mr_prostate_nonrigid"
# print(PROJECT_DIR)  #  demos/classical_mr_prostate_nonrigid

# os.chdir(PROJECT_DIR)

# DATA_PATH = "dataset"
# FILE_PATH = os.path.join(DATA_PATH, "demo2.h5")

# print(FILE_PATH)  # dataset/demo2.h5
# ## registration parameters
# image_loss_name = "lncc"
# deform_loss_name = "bending"
# weight_deform_loss = 10
# learning_rate = 0.1
# total_iter = int(3000)


# ## load images
# if not os.path.exists(DATA_PATH):
#     raise ("Download the data using demo_data.py script")
# if not os.path.exists(FILE_PATH):
#     raise ("Download the data using demo_data.py script")

# fid = h5py.File(FILE_PATH, "r")  # read only, the file must exist

# print(type(fid))
# print(fid)  # <HDF5 file "demo2.h5" (mode r)>


# print(
#     "fid image0", fid["image0"]
# )  # <HDF5 dataset "image0": shape (32, 128, 128), type "<i2">
# print(
#     "fid image1", fid["image1"]
# )  # <HDF5 dataset "image1": shape (32, 128, 128), type "<i2">

# moving_image = tf.cast(tf.expand_dims(fid["image0"], axis=0), dtype=tf.float32)

# print(type(moving_image))  # <class 'tensorflow.python.framework.ops.EagerTensor'>
# print(moving_image.shape)  #  (1, 32, 128, 128)

# fixed_image = tf.cast(tf.expand_dims(fid["image1"], axis=0), dtype=tf.float32)
# print(type(fixed_image))  # <class 'tensorflow.python.framework.ops.EagerTensor'>
# print(fixed_image.shape)  # (1, 32, 128, 128)


# # ddf as trainable weights
# fixed_image_size = fixed_image.shape
# initialiser = tf.random_normal_initializer(mean=0, stddev=1e-3)
# var_ddf = tf.Variable(initialiser(fixed_image_size + [3]), name="ddf", trainable=True)

# print("fixed image size", fixed_image_size)  # (1, 32, 128, 128)
# print("plus three", fixed_image_size + [3])  # (1,32,128,128,3)

# print(
#     initialiser
# )  # <tensorflow.python.ops.init_ops_v2.RandomNormal object at 0x7f6d8082f250>
# # print(  initialiser(fixed_image_size + [3])   )

# # following is not related with the codes
# # use the initialiser to random generate numbers
# print(initialiser((2, 1)))
# notused_one = initialiser((1, 3))
# print(notused_one.shape)  # (1, 3)
# print(notused_one)
# notused_two = initialiser((3, 1))
# print(notused_two.shape)  # (3, 1)
# print(notused_two)
# # the above is not related with the codes.


# var_ddf = tf.Variable(initialiser(fixed_image_size + [3]), name="ddf", trainable=True)

# print(var_ddf.shape)  # (1, 32, 128, 128, 3)
# print(
#     type(var_ddf)
# )  # <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>

# fixed_image_size1 = fixed_image_size[1:4]
# fixed_image_size2 = fixed_image_size[2:4]
# print(fixed_image_size1)  # (32, 128, 128)
# print(fixed_image_size2)  # (128, 128)
# print(
#     type(fixed_image_size1)
# )  # <class 'tensorflow.python.framework.tensor_shape.TensorShape'>

# warping = layer.Warping(fixed_image_size1)
# print(warping)  # <deepreg.model.layer.Warping object at 0x7ff1b4e10e90>


# optimiser = tf.optimizers.Adam(learning_rate)


# ## optimisation
# @tf.function
# def train_step(warper, weights, optimizer, mov, fix):
#     """
#     Train step function for backprop using gradient tape

#     :param warper: warping function returned from layer.Warping
#     :param weights: trainable ddf [1, f_dim1, f_dim2, f_dim3, 3]
#     :param optimizer: tf.optimizers
#     :param mov: moving image [1, m_dim1, m_dim2, m_dim3]
#     :param fix: fixed image [1, f_dim1, f_dim2, f_dim3]
#     :return:
#         loss: overall loss to optimise
#         loss_image: image dissimilarity
#         loss_deform: deformation regularisation
#     """
#     with tf.GradientTape() as tape:
#         pred = warper(inputs=[weights, mov])
#         loss_image = image_loss.dissimilarity_fn(
#             y_true=fix, y_pred=pred, name=image_loss_name
#         )
#         loss_deform = deform_loss.local_displacement_energy(weights, deform_loss_name)
#         loss = loss_image + weight_deform_loss * loss_deform
#     gradients = tape.gradient(loss, [weights])
#     optimizer.apply_gradients(zip(gradients, [weights]))
#     return loss, loss_image, loss_deform


# # for one single iteration
# loss_opt, loss_image_opt, loss_deform_opt = train_step(
#     warping, var_ddf, optimiser, moving_image, fixed_image
# )

# print(loss_opt)  # tf.Tensor([-0.17876787], shape=(1,), dtype=float32)
# print(loss_image_opt)  # tf.Tensor([-0.17879407], shape=(1,), dtype=float32)
# print(loss_deform_opt)  # tf.Tensor([2.619847e-06], shape=(1,), dtype=float32)

# warped_moving_image = warping(inputs=[var_ddf, moving_image])
# idx_slices = [int(5 + x * 5) for x in range(int(fixed_image_size[3] / 5) - 1)]
# print(
#     idx_slices
# )  # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]

# nIdx = len(idx_slices)
# print(nIdx)  # 24
# print(plt.get_backend())  # agg
# for idx in range(len(idx_slices)):
#     print(idx)  # from 0 to 23 [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
#     # ,20,21,22,23]

# plt.figure()

# for idx in range(len(idx_slices)):
#     axs = plt.subplot(nIdx, 3, 3 * idx + 1)
#     axs.imshow(moving_image[0, ..., idx_slices[idx]], cmap="gray")
#     axs.axis("off")
#     axs = plt.subplot(nIdx, 3, 3 * idx + 2)
#     axs.imshow(fixed_image[0, ..., idx_slices[idx]], cmap="gray")
#     axs.axis("off")
#     axs = plt.subplot(nIdx, 3, 3 * idx + 3)
#     axs.imshow(warped_moving_image[0, ..., idx_slices[idx]], cmap="gray")
#     axs.axis("off")
# plt.ion()  # Turn the interactive mode on.
# plt.show()


# # do the warping using the optimized
# moving_label = tf.cast(tf.expand_dims(fid["label0"], axis=0), dtype=tf.float32)
# print(moving_label)  # shape = (1, 32, 128, 128), dtype = float32
# print(moving_label.shape)  # (1, 32, 128, 128)

# fixed_label = tf.cast(tf.expand_dims(fid["label1"], axis=0), dtype=tf.float32)
# print(fixed_label)  # shape=(1, 32, 128, 128), dtype=float32)
# print(fixed_label.shape)  # (1, 32, 128, 128)

# warped_moving_label = warping(inputs=[var_ddf, moving_label])
# print(warped_moving_label)  # shape=(1, 32, 128, 128), dtype=float32)
# print(warped_moving_label.shape)  # (1, 32, 128, 128)

# # let us look into what is going on

# print(nIdx)
# idx = 0
# plt.figure()
# axs = plt.subplot(nIdx, 3, 3 * idx + 1)
# axs.imshow(moving_label[0, ..., idx_slices[idx]], cmap="gray")
# axs.axis("off")
# plt.ion()
# plt.show()


# # axs = plt.subplot(nIdx, 3, 3 * idx + 2)
# # axs.imshow(fixed_label[0, ..., idx_slices[idx]], cmap="gray")
# # axs.axis("off")
# # axs = plt.subplot(nIdx, 3, 3 * idx + 3)
# # axs.imshow(warped_moving_label[0, ..., idx_slices[idx]], cmap="gray")
# # axs.axis("off")
