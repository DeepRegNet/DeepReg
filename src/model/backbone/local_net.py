import tensorflow as tf

from src.model import layer as layer, layer_util as layer_util


class LocalModel(tf.keras.Model):
    def __init__(self,
                 moving_image_size, fixed_image_size,
                 num_channel_initial, ddf_levels=None, **kwargs):
        """
        image is encoded gradually, i from level 0 to E
        then it is decoded gradually, j from level E to D
        some of the decoded level are used for generating ddf

        so ddf_levels are between [0, E] with E = max(ddf_levels) and D = min(ddf_levels)

        :param moving_image_size: [m_dim1, m_dim2, m_dim3]
        :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
        :param num_channel_initial:
        :param ddf_levels:
        :param kwargs:
        """
        super(LocalModel, self).__init__(**kwargs)

        # save parameters
        self._moving_image_size = moving_image_size
        self._fixed_image_size = fixed_image_size
        self._ddf_levels = [0, 1, 2, 3, 4] if ddf_levels is None else ddf_levels
        self._ddf_max_level = max(self._ddf_levels)  # E
        self._ddf_min_level = min(self._ddf_levels)  # D

        # init layer variables
        self._resize3d = layer.Resize3d(size=fixed_image_size)

        nc = [num_channel_initial * (2 ** level) for level in range(self._ddf_max_level + 1)]  # level 0 to E
        self._downsample_blocks = [layer.DownSampleResnetBlock(filters=nc[i], kernel_size=7 if i == 0 else 3)
                                   for i in range(self._ddf_max_level)]  # level 0 to E-1
        self._conv3d_block = layer.Conv3dBlock(filters=nc[-1])  # level E

        self._upsample_blocks = [layer.UpSampleResnetBlock(nc[level]) for level in
                                 range(self._ddf_max_level - 1, self._ddf_min_level - 1, -1)]  # level D to E-1

        self._ddf_summands = [layer.Conv3dWithResize(output_shape=fixed_image_size) for _ in self._ddf_levels]

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: [moving_image, fixed_image]
                        moving_image.shape = [batch, m_dim1, m_dim2, m_dim3]
                        fixed_image.shape = [batch, f_dim1, f_dim2, f_dim3]
        :param training:
        :param mask:
        :return:
        """
        layer_util.check_inputs(inputs, 2, "LocalModel")

        moving_image = tf.expand_dims(inputs[0], axis=4)  # [batch, m_dim1, m_dim2, m_dim3, 1]
        fixed_image = tf.expand_dims(inputs[1], axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 1]
        images = tf.concat([self._resize3d(inputs=moving_image), fixed_image],
                           axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 2]

        # down sample from level 0 to E
        encoded = []  # outputs used for decoding, encoded[i] corresponds to level i, stored only 0 to E-1
        h = images
        for level in range(self._ddf_max_level):  # level 0 to E - 1
            h, hc = self._downsample_blocks[level](inputs=h, training=training)
            encoded.append(hc)
        hm = self._conv3d_block(inputs=h, training=training)  # level E of encoding/decoding

        # up sample from level E to D
        decoded = [hm]  # level E
        for idx, level in enumerate(range(self._ddf_max_level - 1, self._ddf_min_level - 1, -1)):  # level E-1 to D
            hm = self._upsample_blocks[idx](inputs=[hm, encoded[level]], training=training)
            decoded.append(hm)

        ddf = tf.reduce_sum(tf.stack([self._ddf_summands[idx](inputs=decoded[self._ddf_max_level - level])
                                      for idx, level in enumerate(self._ddf_levels)], axis=5), axis=5)
        return ddf
