import tensorflow as tf

from src.model import layer as layer


class UNet(tf.keras.Model):
    def __init__(self,
                 moving_image_size, fixed_image_size,
                 num_channel_initial, depth, pooling=True, concat_skip=False, **kwargs):
        """
        :param moving_image_size: [m_dim1, m_dim2, m_dim3]
        :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
        :param num_channel_initial:
        :param depth: input is at level 0, bottom is at level depth
        :param pooling: true if use pooling to down sample
        :param kwargs:
        """
        super(UNet, self).__init__(**kwargs)

        # save parameters
        self._fixed_image_size = fixed_image_size

        # init layer variables
        nc = [num_channel_initial * (2 ** d) for d in range(depth + 1)]

        self._num_channel_initial = num_channel_initial
        self._depth = depth
        self._resize3d = layer.Resize3d(size=fixed_image_size)
        self._downsample_blocks = [layer.DownSampleResnetBlock(filters=nc[d], pooling=pooling)
                                   for d in range(depth)]
        self._bottom_conv3d = layer.Conv3dBlock(filters=nc[depth])
        self._bottom_res3d = layer.Residual3dBlock(filters=nc[depth])
        self._upsample_blocks = [layer.UpSampleResnetBlock(filters=nc[d], concat=concat_skip)
                                 for d in range(depth)]
        self._output_conv3d = layer.Conv3dWithResize(output_shape=fixed_image_size, filters=3)

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: [moving_image, fixed_image]
                        moving_image.shape = [batch, m_dim1, m_dim2, m_dim3]
                        fixed_image.shape = [batch, f_dim1, f_dim2, f_dim3]
        :param training:
        :param mask:
        :return:
        """

        moving_image = tf.expand_dims(inputs[0], axis=4)  # [batch, m_dim1, m_dim2, m_dim3, 1]
        fixed_image = tf.expand_dims(inputs[1], axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 1]
        down_sampled = tf.concat([self._resize3d(inputs=moving_image), fixed_image],
                                 axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 2]

        # down sample
        skips = []
        for d in range(self._depth):  # level 0 to D-1
            down_sampled, skip = self._downsample_blocks[d](inputs=down_sampled, training=training)
            skips.append(skip)

        # bottom, level D
        up_sampled = self._bottom_res3d(inputs=self._bottom_conv3d(inputs=down_sampled,
                                                                   training=training),
                                        training=training)

        # up sample, level D-1 to 0
        for d in range(self._depth - 1, -1, -1):
            up_sampled = self._upsample_blocks[d](inputs=[up_sampled, skips[d]], training=training)

        # output
        ddf = self._output_conv3d(inputs=up_sampled)
        return ddf
