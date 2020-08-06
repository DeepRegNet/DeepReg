# import os

# import h5py
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# import tensorflow as tf

# import deepreg.model.layer as layer
# import deepreg.model.loss.deform as deform_loss
# import deepreg.model.loss.image as image
# import deepreg.model.loss.image as image_loss

# MAIN_PATH = os.getcwd()
# print(MAIN_PATH)
# PROJECT_DIR = r"demos/classical_mr_prostate_nonrigid"
# os.chdir(PROJECT_DIR)


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


# fixed_image_size = fixed_image.shape
# initialiser = tf.random_normal_initializer(mean=0, stddev=1e-3)
# var_ddf = tf.Variable(initialiser(fixed_image_size + [3]), name="ddf", trainable=True)

# print(fixed_image_size)
# print(initialiser)
# print(initialiser.shape)


# # tensor_true1 = np.zeros((2, 1, 2, 3))
# # tensor_pred1 = 0.6 * np.ones((2, 1, 2, 3))
# # tensor_true1 = tf.convert_to_tensor(tensor_true1, dtype=tf.float32)
# # tensor_pred1 = tf.convert_to_tensor(tensor_pred1, dtype=tf.float32)
# # image.dissimilarity_fn(tensor_true1, tensor_pred1, "some random string that isn't ssd or lncc")


# # EPS = 1.0e-6

# # main_path = os.getcwd()
# # print(main_path)   # /home/linux-min/DeepReg/test/unit
# # os.chdir(main_path)


# # #tensor_true = np.array(range(12)).reshape((2, 1, 2, 3))

# # tensor_true = np.zeros((2, 1, 2, 3))
# # tensor_pred = 0.6 * np.ones((2, 1, 2, 3))
# # tensor_true = tf.convert_to_tensor(tensor_true)
# # tensor_pred = tf.convert_to_tensor(tensor_pred)
# # tensor_true = tf.cast(tensor_true, dtype=tf.float32)
# # tensor_pred = tf.cast(tensor_pred, dtype=tf.float32)
# # # tensor_true = np.array(tensor_true)
# # # tensor_pred = np.array(tensor_pred)
# # print(type(tensor_true[0, 0, 0, 0]))
# # print(type(tensor_pred[0, 0, 0, 0]))

# # print(type(np.ones((2, 1, 2, 3, 2)) ))
# # print(type(np.zeros((2, 1, 2, 3, 2)) ))
# # # print(type(tensor_true[0, 0, 0, 0]))

# # name_ncc = "lncc"
# # get_ncc = image.dissimilarity_fn(tensor_true,tensor_pred, name_ncc)

# # print(get_ncc)
# # # print(name_ncc)
# # name_ssd = "ssd"
# # get_ssd = image.dissimilarity_fn(tensor_true, tensor_pred, name_ssd)
# # print('get-ssd',get_ssd)


# # # get_ncc = image.local_normalized_cross_correlation(tensor_true_expand_np,tensor_pred_expand_np)
# # #name_notnccssd = "notlnccssd"
# # # get_notnccssd = image.dissimilarity_fn(tensor_true, tensor_pred, name_notnccssd)
# # # print(get_notnccssd)


# # # tensor_true_expand = tf.expand_dims(tensor_true, axis=4)
# # # tensor_pred_expand = tf.expand_dims(tensor_pred, axis=4)
# # # tensor_true_expand_np = np.array(tensor_true_expand)
# # # tensor_pred_expand_np = np.array(tensor_pred_expand)
