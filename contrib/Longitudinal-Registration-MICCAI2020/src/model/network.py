import src.model.layer as layer
import src.model.layer_util as layer_util
import src.model.loss as loss
import tensorflow as tf


class LocalModel(tf.keras.Model):
    def __init__(
        self,
        moving_image_size,
        fixed_image_size,
        num_channel_initial,
        ddf_levels=None,
        **kwargs,
    ):
        super(LocalModel, self).__init__(**kwargs)

        # save parameters
        self._moving_image_size = moving_image_size
        self._fixed_image_size = fixed_image_size
        self._ddf_levels = [0, 1, 2, 3, 4] if ddf_levels is None else ddf_levels
        self._ddf_min_level = min(self._ddf_levels)

        # init layer variables
        self._resize3d = layer.Resize3d(size=fixed_image_size)

        nc = [num_channel_initial * (2 ** i) for i in range(5)]

        self._downsample_block0 = layer.DownSampleResnetBlock(
            filters=nc[0], kernel_size=7
        )
        self._downsample_block1 = layer.DownSampleResnetBlock(filters=nc[1])
        self._downsample_block2 = layer.DownSampleResnetBlock(filters=nc[2])
        self._downsample_block3 = layer.DownSampleResnetBlock(filters=nc[3])
        self._conv3d_block4 = layer.Conv3dBlock(filters=nc[4])

        self._upsample_block3 = (
            layer.UpSampleResnetBlock(nc[3]) if self._ddf_min_level < 4 else None
        )
        self._upsample_block2 = (
            layer.UpSampleResnetBlock(nc[2]) if self._ddf_min_level < 3 else None
        )
        self._upsample_block1 = (
            layer.UpSampleResnetBlock(nc[1]) if self._ddf_min_level < 2 else None
        )
        self._upsample_block0 = (
            layer.UpSampleResnetBlock(nc[0]) if self._ddf_min_level < 1 else None
        )

        self._ddf_summands = [
            layer.DDFSummand(output_shape=fixed_image_size) for _ in self._ddf_levels
        ]

    def call(self, inputs, training=None, mask=None):
        layer_util.check_inputs(inputs, 2, "LocalModel")

        moving_image, fixed_image = inputs[0], inputs[1]
        moving_image = tf.expand_dims(moving_image, axis=4)
        fixed_image = tf.expand_dims(fixed_image, axis=4)
        images = tf.concat(
            [self._resize3d(inputs=moving_image), fixed_image], axis=4
        )  # [batch, f_dim1, f_dim2, f_dim3, 2]

        h0, hc0 = self._downsample_block0(inputs=images, training=training)
        h1, hc1 = self._downsample_block1(inputs=h0, training=training)
        h2, hc2 = self._downsample_block2(inputs=h1, training=training)
        h3, hc3 = self._downsample_block3(inputs=h2, training=training)
        hm = [self._conv3d_block4(inputs=h3, training=training)]

        hm += (
            [self._upsample_block3(inputs=[hm[0], hc3], training=training)]
            if self._ddf_min_level < 4
            else []
        )
        hm += (
            [self._upsample_block2(inputs=[hm[1], hc2], training=training)]
            if self._ddf_min_level < 3
            else []
        )
        hm += (
            [self._upsample_block1(inputs=[hm[2], hc1], training=training)]
            if self._ddf_min_level < 2
            else []
        )
        hm += (
            [self._upsample_block0(inputs=[hm[3], hc0], training=training)]
            if self._ddf_min_level < 1
            else []
        )

        ddf = tf.reduce_sum(
            tf.stack(
                [self._ddf_summands[i](inputs=hm[4 - i]) for i in self._ddf_levels],
                axis=5,
            ),
            axis=5,
        )

        # for i in self._ddf_levels:
        #     print(i, hm[4 - i].shape)
        #
        # ddf = tf.reduce_sum(tf.stack([self._ddf_summands[i](inputs=hm[4 - i][..., :int(hm[4 - i].shape[-1]/2)]) for i in self._ddf_levels], axis=5),
        #                     axis=5)
        # ddf2 = tf.reduce_sum(tf.stack([self._ddf_summands[i](inputs=hm[4 - i][..., int(hm[4 - i].shape[-1]/2):]) for i in self._ddf_levels], axis=5),
        #                     axis=5)

        # print('images:', images.shape)
        # print('h0:', h0.shape, hc0.shape)
        # print('h1:', h1.shape, hc1.shape)
        # print('h2:', h2.shape, hc2.shape)
        # print('h3:', h3.shape, hc3.shape)
        # print('hm:', hm[0].shape)
        # print('ddf:', ddf.shape)
        # print('ddf2:', ddf2.shape)

        return ddf, hm[0]


def build_model(
    moving_image_size,
    fixed_image_size,
    batch_size,
    num_channel_initial,
    ddf_levels,
    ddf_energy_type,
):
    # inputs
    moving_image = tf.keras.Input(
        shape=(*moving_image_size,), batch_size=batch_size, name="moving_image"
    )
    fixed_image = tf.keras.Input(
        shape=(*fixed_image_size,), batch_size=batch_size, name="fixed_image"
    )
    moving_label = tf.keras.Input(
        shape=(*moving_image_size,), batch_size=batch_size, name="moving_label"
    )

    # backbone
    local_model = LocalModel(
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        num_channel_initial=num_channel_initial,
        ddf_levels=ddf_levels,
    )

    # ddf
    ddf, hm_0 = local_model(inputs=[moving_image, fixed_image])

    # prediction
    warped_moving_image = layer.Warping(fixed_image_size=fixed_image_size)(
        [ddf, moving_image]
    )  # modified
    warped_moving_label = layer.Warping(fixed_image_size=fixed_image_size)(
        [ddf, moving_label]
    )

    # build model
    registration_model = tf.keras.Model(
        inputs=[moving_image, fixed_image, moving_label],
        outputs=[warped_moving_image, warped_moving_label, hm_0],
        name="RegModel",
    )  # modified

    # add regularization loss
    bending_loss = tf.reduce_mean(
        loss.local_displacement_energy(ddf, ddf_energy_type, 0.5)
    )
    # sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    # mmd_loss = loss.loss_mmd(hm_0[0, ...], hm_0[1, ...], sigmas)
    registration_model.add_loss(bending_loss)
    # registration_model.add_loss(mmd_loss)
    return registration_model
