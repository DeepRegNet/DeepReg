import tensorflow as tf

from deepreg.model import layer as layer


class UNet(tf.keras.Model):
    def __init__(
        self,
        image_size,
        out_channels,
        num_channel_initial,
        depth,
        out_kernel_initializer,
        out_activation,
        pooling=True,
        concat_skip=False,
        **kwargs,
    ):
        """
        :param image_size: [f_dim1, f_dim2, f_dim3]
        :param out_channels: number of channels for the output
        :param num_channel_initial:
        :param depth: input is at level 0, bottom is at level depth
        :param out_kernel_initializer:
        :param out_activation:
        :param pooling: true if use pooling to down sample
        :param kwargs:
        """
        super(UNet, self).__init__(**kwargs)

        # init layer variables
        nc = [num_channel_initial * (2 ** d) for d in range(depth + 1)]

        self._num_channel_initial = num_channel_initial
        self._depth = depth
        self._downsample_blocks = [
            layer.DownSampleResnetBlock(filters=nc[d], pooling=pooling)
            for d in range(depth)
        ]
        self._bottom_conv3d = layer.Conv3dBlock(filters=nc[depth])
        self._bottom_res3d = layer.Residual3dBlock(filters=nc[depth])
        self._upsample_blocks = [
            layer.UpSampleResnetBlock(filters=nc[d], concat=concat_skip)
            for d in range(depth)
        ]
        self._output_conv3d = layer.Conv3dWithResize(
            output_shape=image_size,
            filters=out_channels,
            kernel_initializer=out_kernel_initializer,
            activation=out_activation,
        )

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: shape = [batch, f_dim1, f_dim2, f_dim3, in_channels]
        :param training:
        :param mask:
        :return: shape = [batch, f_dim1, f_dim2, f_dim3, out_channels]
        """

        down_sampled = inputs

        # down sample
        skips = []
        for d in range(self._depth):  # level 0 to D-1
            down_sampled, skip = self._downsample_blocks[d](
                inputs=down_sampled, training=training
            )
            skips.append(skip)

        # bottom, level D
        up_sampled = self._bottom_res3d(
            inputs=self._bottom_conv3d(inputs=down_sampled, training=training),
            training=training,
        )

        # up sample, level D-1 to 0
        for d in range(self._depth - 1, -1, -1):
            up_sampled = self._upsample_blocks[d](
                inputs=[up_sampled, skips[d]], training=training
            )

        # output
        output = self._output_conv3d(inputs=up_sampled)
        return output
