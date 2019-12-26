import tensorflow as tf

import src.model.layer as layer
import src.model.util as util


class LocalModel(tf.keras.Model):
    def __init__(self, batch_size, num_channel_initial, ddf_levels=None, **kwargs):
        super(LocalModel, self).__init__()

        # save parameters
        self._batch_size = batch_size
        self._ddf_levels = [0, 1, 2, 3, 4] if ddf_levels is None else ddf_levels

        # init layer variables
        self._resize3d = None

        nc = [num_channel_initial * (2 ** i) for i in range(5)]
        self._downsample_resnet_block0 = layer.DownSampleResnetBlock(filters=nc[0], kernel_size=7)
        self._downsample_resnet_block1 = layer.DownSampleResnetBlock(filters=nc[1])
        self._downsample_resnet_block2 = layer.DownSampleResnetBlock(filters=nc[2])
        self._downsample_resnet_block3 = layer.DownSampleResnetBlock(filters=nc[3])
        self._conv3d_block4 = layer.Conv3dBlock(filters=nc[4])

        upsample_resnet_blocks = []
        min_level = min(self._ddf_levels)
        for i in range(4, 0, -1):
            if min_level < i:
                upsample_resnet_blocks.append(layer.UpSampleResnetBlock(filters=nc[i - 1]))
        self._upsample_resnet_blocks = upsample_resnet_blocks
        self._ddf_summands = None
        self._grid_ref = None

    def build(self, input_shape):
        # sanity check
        if not (isinstance(input_shape, list) or isinstance(input_shape, tuple)):
            raise ValueError("LocalModel accepts an input as a list of tensors [moving_image, fixed_image]")
        if len(input_shape) != 3:
            raise ValueError("LocalModel accepts an input as a list of tensors [moving_image, fixed_image]")
        super(LocalModel, self).build(input_shape)

        moving_image_size = input_shape[0][1:4]
        fixed_image_size = input_shape[1][1:4]
        self._moving_image_size = moving_image_size
        self._fixed_image_size = fixed_image_size
        self._resize3d = layer.Resize3d(size=fixed_image_size)
        self._ddf_summands = [layer.DDFSummand(output_shape=fixed_image_size) for _ in self._ddf_levels]
        self._grid_ref = util.get_reference_grid(grid_size=fixed_image_size)

    def call(self, inputs, training=None, mask=None):
        # sanity check
        if not isinstance(inputs, list):
            raise ValueError("LocalModel accepts an input as a list of tensors [moving_image, fixed_image]")
        if len(inputs) != 3:
            raise ValueError("LocalModel accepts an input as a list of tensors [moving_image, fixed_image]")

        moving_image, fixed_image, moving_label = inputs[0], inputs[1], inputs[2]

        moving_image = tf.reshape(moving_image, [self._batch_size] + self._moving_image_size + [1])
        fixed_image = tf.reshape(fixed_image, [self._batch_size] + self._fixed_image_size + [1])
        moving_label = tf.reshape(moving_label, [self._batch_size] + self._moving_image_size + [1])
        # moving_image = tf.expand_dims(moving_image, axis=4)  # [batch, m_dim1, m_dim2, m_dim3, 1]
        # fixed_image = tf.expand_dims(fixed_image, axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 1]
        # moving_label = tf.expand_dims(moving_label, axis=4)  # [batch, m_dim1, m_dim2, m_dim3, 1]
        inputs = tf.concat([self._resize3d(inputs=moving_image), fixed_image],
                           axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 2]

        h0, hc0 = self._downsample_resnet_block0(inputs=inputs, training=training)
        h1, hc1 = self._downsample_resnet_block1(inputs=h0, training=training)
        h2, hc2 = self._downsample_resnet_block2(inputs=h1, training=training)
        h3, hc3 = self._downsample_resnet_block3(inputs=h2, training=training)
        hm = [self._conv3d_block4(inputs=h3, training=training)]
        hcs = [hc0, hc1, hc2, hc3]

        for i in range(len(self._upsample_resnet_blocks)):
            hm.append(self._upsample_resnet_blocks[i](inputs=[hm[i], hcs[3 - i]], training=training))

        ddf = tf.reduce_sum(tf.stack([self._ddf_summands[i](inputs=hm[4 - i]) for i in self._ddf_levels], axis=5),
                            axis=5)

        grid_warped = self._grid_ref + ddf
        warped_moving_label = util.warp_moving(moving_image_or_label=moving_label, grid_warped=grid_warped)
        return warped_moving_label
