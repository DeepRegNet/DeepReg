import tensorflow as tf

from src.model import layer as layer, layer_util as layer_util


class LocalModel(tf.keras.Model):
    def __init__(self,
                 moving_image_size, fixed_image_size,
                 num_channel_initial, ddf_levels=None, **kwargs):
        super(LocalModel, self).__init__(**kwargs)

        # save parameters
        self._moving_image_size = moving_image_size
        self._fixed_image_size = fixed_image_size
        self._ddf_levels = [0, 1, 2, 3, 4] if ddf_levels is None else ddf_levels
        self._ddf_min_level = min(self._ddf_levels)

        # init layer variables
        self._resize3d = layer.Resize3d(size=fixed_image_size)

        nc = [num_channel_initial * (2 ** i) for i in range(5)]
        self._downsample_block0 = layer.DownSampleResnetBlock(filters=nc[0], kernel_size=7)
        self._downsample_block1 = layer.DownSampleResnetBlock(filters=nc[1])
        self._downsample_block2 = layer.DownSampleResnetBlock(filters=nc[2])
        self._downsample_block3 = layer.DownSampleResnetBlock(filters=nc[3])
        self._conv3d_block4 = layer.Conv3dBlock(filters=nc[4])

        self._upsample_block3 = layer.UpSampleResnetBlock(nc[3]) if self._ddf_min_level < 4 else None
        self._upsample_block2 = layer.UpSampleResnetBlock(nc[2]) if self._ddf_min_level < 3 else None
        self._upsample_block1 = layer.UpSampleResnetBlock(nc[1]) if self._ddf_min_level < 2 else None
        self._upsample_block0 = layer.UpSampleResnetBlock(nc[0]) if self._ddf_min_level < 1 else None

        self._ddf_summands = [layer.DDFSummand(output_shape=fixed_image_size) for _ in self._ddf_levels]

    def call(self, inputs, training=None, mask=None):
        layer_util.check_inputs(inputs, 2, "LocalModel")

        moving_image, fixed_image = inputs[0], inputs[1]
        moving_image = tf.expand_dims(moving_image, axis=4)
        fixed_image = tf.expand_dims(fixed_image, axis=4)
        images = tf.concat([self._resize3d(inputs=moving_image), fixed_image],
                           axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 2]

        h0, hc0 = self._downsample_block0(inputs=images, training=training)
        h1, hc1 = self._downsample_block1(inputs=h0, training=training)
        h2, hc2 = self._downsample_block2(inputs=h1, training=training)
        h3, hc3 = self._downsample_block3(inputs=h2, training=training)
        hm = [self._conv3d_block4(inputs=h3, training=training)]

        hm += [self._upsample_block3(inputs=[hm[0], hc3], training=training)] if self._ddf_min_level < 4 else []
        hm += [self._upsample_block2(inputs=[hm[1], hc2], training=training)] if self._ddf_min_level < 3 else []
        hm += [self._upsample_block1(inputs=[hm[2], hc1], training=training)] if self._ddf_min_level < 2 else []
        hm += [self._upsample_block0(inputs=[hm[3], hc0], training=training)] if self._ddf_min_level < 1 else []

        ddf = tf.reduce_sum(tf.stack([self._ddf_summands[i](inputs=hm[4 - i]) for i in self._ddf_levels], axis=5),
                            axis=5)
        return ddf