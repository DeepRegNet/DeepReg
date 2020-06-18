import tensorflow as tf

from deepreg.model import layer as layer
from deepreg.model import layer_util as layer_util


class GlobalNet(tf.keras.Model):
    def __init__(self,
                 image_size, out_channels,
                 num_channel_initial, extract_levels,
                 out_kernel_initializer, out_activation,
                 **kwargs):
        """
        image is encoded gradually, i from level 0 to E
        then a densely-connected layer outputs an affine
        transformation.

        :param out_channels: number of channels for the extractions
        :param num_channel_initial:
        :param extract_levels:
        :param out_kernel_initializer:
        :param out_activation:
        :param kwargs:
        """
        super(GlobalNet, self).__init__(**kwargs)

        # save parameters
        self._extract_levels = extract_levels
        self._extract_max_level = max(self._extract_levels)  # E
        self.reference_grid = layer_util.get_reference_grid(image_size)
        self.transform_initial = tf.constant_initializer(value=[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.])

        # init layer variables
        nc = [num_channel_initial * (2 ** level) for level in range(self._extract_max_level + 1)]  # level 0 to E
        self._downsample_blocks = [layer.DownSampleResnetBlock(filters=nc[i], kernel_size=7 if i == 0 else 3)
                                   for i in range(self._extract_max_level)]  # level 0 to E-1
        self._conv3d_block = layer.Conv3dBlock(filters=nc[-1])  # level E
        self._dense_layer = layer.Dense(units=12, bias_initializer=self.transform_initial) 
        self._reshape = tf.keras.layers.Reshape(target_shape=(4, 3))

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: shape = [batch, f_dim1, f_dim2, f_dim3, ch]
        :param training:
        :param mask:
        :return:
        """
        # down sample from level 0 to E
        h = inputs
        for level in range(self._extract_max_level):  # level 0 to E - 1
            h, _ = self._downsample_blocks[level](inputs=h, training=training)
        hm = self._conv3d_block(inputs=h, training=training)  # level E of encoding

        # predict affine parameters theta of shape = [batch, 4, 3]
        theta = self._dense_layer(hm)
        theta = self._reshape(theta)

        # warp the reference grid with affine parameters to output a ddf
        grid_warped = layer_util.warp_grid(self.reference_grid, theta)
        output = grid_warped - self.reference_grid
        return output
