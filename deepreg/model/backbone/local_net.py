#  coding=utf-8

"""
Module to build LocalNet class based on:

Hu, Yipeng, et al. "Weakly-supervised convolutional neural networks for multimodal image registration." Medical image analysis 49 (2018): 1-13.
https://doi.org/10.1016/j.media.2018.07.002
"""

import tensorflow as tf

from deepreg.model import layer


class LocalNet(tf.keras.Model):
    """
    Builds LocalNet for image registration based on
    Y. Hu et al.,
    "Label-driven weakly-supervised learning for multimodal
    deformable image registration,"
    (ISBI 2018), pp. 1070-1074.
    """

    def __init__(
        self,
        image_size,
        out_channels,
        num_channel_initial,
        extract_levels,
        out_kernel_initializer,
        out_activation,
        **kwargs,
    ):
        """
        Initialising LocalNet.
        Image is enum_channelsoded gradually, i from level 0 to E,
        then it is decoded gradually, j from level E to D
        Some of the decoded levels are used for generating extractions

        So, extract_levels are between [0, E] with E = max(extract_levels),
        and D = min(extract_levels).

        :param out_channels: int, number of channels for the extractions
        :param num_channel_initial: int, number of initial channels.
        :param extract_levels: int, number of extraction levels.
        :param out_kernel_initializer: str, initialiser to use for kernels.
        :param out_activation: str, activation to use at end layer.
        :param kwargs:
        """
        super(LocalNet, self).__init__(**kwargs)

        # save parameters
        self._extract_levels = extract_levels
        self._extract_max_level = max(self._extract_levels)  # E
        self._extract_min_level = min(self._extract_levels)  # D

        # init layer variables

        num_channels = [
            num_channel_initial * (2 ** level)
            for level in range(self._extract_max_level + 1)
        ]  # level 0 to E
        self._downsample_blocks = [
            layer.DownSampleResnetBlock(
                filters=num_channels[i], kernel_size=7 if i == 0 else 3
            )
            for i in range(self._extract_max_level)
        ]  # level 0 to E-1
        self._conv3d_block = layer.Conv3dBlock(filters=num_channels[-1])  # level E

        self._upsample_blocks = [
            layer.LocalNetUpSampleResnetBlock(num_channels[level])
            for level in range(
                self._extract_max_level - 1, self._extract_min_level - 1, -1
            )
        ]  # level D to E-1

        self._extract_layers = [
            # if kernels are not initialized by zeros, with init NN, extract may be too large
            layer.Conv3dWithResize(
                output_shape=image_size,
                filters=out_channels,
                kernel_initializer=out_kernel_initializer,
                activation=out_activation,
            )
            for _ in self._extract_levels
        ]

    def call(self, inputs, training=None, mask=None):
        """
        Build LocalNet graph based on built layers.
        :param inputs: image batch, shape = [batch, f_dim1, f_dim2, f_dim3, ch]
        :param training:
        :param mask:
        :return:
        """

        # down sample from level 0 to E
        enum_channelsoded = []
        # outputs used for decoding, enum_channelsoded[i] corresponds -> level i
        #  stored only 0 to E-1

        h_in = inputs
        for level in range(self._extract_max_level):  # level 0 to E - 1
            h_in, h_channel = self._downsample_blocks[level](
                inputs=h_in, training=training
            )
            enum_channelsoded.append(h_channel)
        h_bottom = self._conv3d_block(
            inputs=h_in, training=training
        )  # level E of enum_channelsoding/decoding

        # up sample from level E to D
        decoded = [h_bottom]  # level E
        for idx, level in enumerate(
            range(self._extract_max_level - 1, self._extract_min_level - 1, -1)
        ):  # level E-1 to D
            h_bottom = self._upsample_blocks[idx](
                inputs=[h_bottom, enum_channelsoded[level]], training=training
            )
            decoded.append(h_bottom)

        # output
        output = tf.reduce_mean(
            tf.stack(
                [
                    self._extract_layers[idx](
                        inputs=decoded[self._extract_max_level - level]
                    )
                    for idx, level in enumerate(self._extract_levels)
                ],
                axis=5,
            ),
            axis=5,
        )
        return output
