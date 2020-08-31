# Custom network

This tutorial shows how to define a new network and add it to DeepReg, using a specific
example for adding a GlobalNet to predict an affine transformation, as opposed to
nonrigid transformation.

For general guidance on making a contribution to DeepReg, see the
[contribution guidelines](../contributing/guide.html).

## Step 1: Create network backbone

The first step is to create a new backbone class, which consists of the neural network
architecture you want to use, and place it in the backbone directory
`deepreg/model/backbone/`. The affine method uses the GlobalNet network architecture
(`deepreg/model/backbone/global_net.py`) from
[Hu et al. 2018](https://ieeexplore.ieee.org/abstract/document/8363756?casa_token=FhpScE4qdoAAAAAA:dJqOru2PqjQCYm-n81fg7lVL5fC7bt6zQHiU6j_EdfIj7Ihm5B9nd7w5Eh0RqPFWLxahwQJ2Xw).
The GlobalNet network has an encoder-only architecture, which is used to predict the
parameters of an affine transformation model, with 12 degrees of freedom.

We recommend using the
[tf.keras API](https://www.tensorflow.org/api_docs/python/tf/keras/Model) to write your
network, by defining the layers of your backbone class in `def __init__()` and the
network's forward pass in `def call()`. Custom DeepReg layers can be found in
`deepreg/model/layer.py`.

    class GlobalNet(tf.keras.Model):
        """
        Builds GlobalNet for image registration based on
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
            Image is encoded gradually, i from level 0 to E.
            Then, a densely-connected layer outputs an affine
            transformation.
            :param out_channels: int, number of channels for the output
            :param num_channel_initial: int, number of initial channels
            :param extract_levels: list, which levels from net to extract
            :param out_activation: str, activation at last layer
            :param out_kernel_initializer: str, which kernel to use as initialiser
            :param kwargs:
            """
            super(GlobalNet, self).__init__(**kwargs)
            # save parameters
            self._extract_levels = extract_levels
            self._extract_max_level = max(self._extract_levels)  # E
            self.reference_grid = layer_util.get_reference_grid(image_size)
            self.transform_initial = tf.constant_initializer(
                value=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            )
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
            self._dense_layer = layer.Dense(
                units=12, bias_initializer=self.transform_initial
            )

        def call(self, inputs, training=None, mask=None):
            """
            Build GlobalNet graph based on built layers.
            :param inputs: image batch, shape = [batch, f_dim1, f_dim2, f_dim3, ch]
            :param training:
            :param mask:
            :return:
            """
            # down sample from level 0 to E
            h_in = inputs
            for level in range(self._extract_max_level):  # level 0 to E - 1
                h_in, _ = self._downsample_blocks[level](inputs=h_in, training=training)
            h_out = self._conv3d_block(
                inputs=h_in, training=training
            )  # level E of encoding
            # predict affine parameters theta of shape = [batch, 4, 3]
            self.theta = self._dense_layer(h_out)
            self.theta = tf.reshape(self.theta, shape=(-1, 4, 3))
            # warp the reference grid with affine parameters to output a ddf
            grid_warped = layer_util.warp_grid(self.reference_grid, self.theta)
            output = grid_warped - self.reference_grid
            return output`

In order to use the backbone network in the DeepReg pipeline, a new option needs to be
added to `build_backbone()` from `deepreg/model/network/util.py`. We use the keyword
"global" here to refer to our `GlobalNet` class and `"affine"` for the method name. This
will allow us to define the backbone network directly in the configuration file.

    def build_backbone(
        image_size: tuple, out_channels: int, model_config: dict, method_name: str
    ) -> tf.keras.Model:
        """
        Backbone model accepts a single input of shape (batch, dim1, dim2, dim3, ch_in)
        and returns a single output of shape (batch, dim1, dim2, dim3, ch_out)
        :param image_size: tuple, dims of image, (dim1, dim2, dim3)
        :param out_channels: int, number of out channels, ch_out
        :param method_name: str, one of ddf | dvf | conditional
        :param model_config: dict, model configuration, returned from parser.yaml.load
        :return: tf.keras.Model
        """
        if not (
            (isinstance(image_size, tuple) or isinstance(image_size, list))
            and len(image_size) == 3
        ):
            raise ValueError(f"image_size must be tuple of length 3, got {image_size}")
        if not (isinstance(out_channels, int) and out_channels >= 1):
            raise ValueError(f"out_channels must be int >=1, got {out_channels}")
        if not (isinstance(model_config, dict) and "backbone" in model_config.keys()):
            raise ValueError(
                f"model_config must be a dict having key 'backbone', got{model_config}"
            )
        if method_name not in ["ddf", "dvf", "conditional", "affine"]:
            raise ValueError(
                "method name has to be one of ddf/dvf/conditional/affine in build_backbone, "
                "got {}".format(method_name)
            )

        if method_name in ["ddf", "dvf"]:
            out_activation = None
            # TODO try random init with smaller number
            out_kernel_initializer = "zeros"  # to ensure small ddf and dvf
        elif method_name in ["conditional"]:
            out_activation = "sigmoid"  # output is probability
            out_kernel_initializer = "glorot_uniform"
        elif method_name in ["affine"]:
            out_activation = None
            out_kernel_initializer = "zeros"
        else:
            raise ValueError("Unknown method name {}".format(method_name))

        if model_config["backbone"] == "local":
            return LocalNet(
                image_size=image_size,
                out_channels=out_channels,
                out_kernel_initializer=out_kernel_initializer,
                out_activation=out_activation,
                **model_config["local"],
            )
        elif model_config["backbone"] == "global":
            return GlobalNet(
                image_size=image_size,
                out_channels=out_channels,
                out_kernel_initializer=out_kernel_initializer,
                out_activation=out_activation,
                **model_config["global"],
            )
        elif model_config["backbone"] == "unet":
            return UNet(
                image_size=image_size,
                out_channels=out_channels,
                out_kernel_initializer=out_kernel_initializer,
                out_activation=out_activation,
                **model_config["unet"],
            )
        else:
            raise ValueError("Unknown model name")

## Step 2: Create network model

We can now create a network model for the affine method in
`deepreg/model/network/affine.py` . We first need to write the model's forward pass,
which makes use of the backbone network class to predict an affine transformation which
will be used to output a dense displacement field (DDF).

    def affine_forward(
        backbone: tf.keras.Model,
        moving_image: tf.Tensor,
        fixed_image: tf.Tensor,
        moving_label: (tf.Tensor, None),
        moving_image_size: tuple,
        fixed_image_size: tuple,
    ):
        """
        Perform the network forward pass
        :param backbone: model architecture object, e.g. model.backbone.local_net
        :param moving_image: tensor of shape (batch, m_dim1, m_dim2, m_dim3)
        :param fixed_image:  tensor of shape (batch, f_dim1, f_dim2, f_dim3)
        :param moving_label: tensor of shape (batch, m_dim1, m_dim2, m_dim3) or None
        :param moving_image_size: tuple like (m_dim1, m_dim2, m_dim3)
        :param fixed_image_size: tuple like (f_dim1, f_dim2, f_dim3)
        :return: tuple(_affine, _ddf, _pred_fixed_image, _pred_fixed_label)
        :return: tuple(affine, ddf, pred_fixed_image, pred_fixed_label, fixed_grid), where
        - affine is the affine transformation matrix predicted by the network (batch, 4, 3)
        - ddf is the dense displacement field of shape (batch, f_dim1, f_dim2, f_dim3, 3)
        - pred_fixed_image is the predicted (warped) moving image of shape (batch, f_dim1, f_dim2, f_dim3)
        - pred_fixed_label is the predicted (warped) moving label of shape (batch, f_dim1, f_dim2, f_dim3)
        - fixed_grid is the grid of shape(f_dim1, f_dim2, f_dim3, 3)
        """

        # expand dims
        # need to be squeezed later for warping
        moving_image = tf.expand_dims(
            moving_image, axis=4
        )  # (batch, m_dim1, m_dim2, m_dim3, 1)
        fixed_image = tf.expand_dims(
            fixed_image, axis=4
        )  # (batch, f_dim1, f_dim2, f_dim3, 1)

        # adjust moving image
        moving_image = layer_util.resize3d(
            image=moving_image, size=fixed_image_size
        )  # (batch, f_dim1, f_dim2, f_dim3, 1)

        # ddf, dvf
        inputs = tf.concat(
            [moving_image, fixed_image], axis=4
        )  # (batch, f_dim1, f_dim2, f_dim3, 2)
        ddf = backbone(inputs=inputs)  # (batch, f_dim1, f_dim2, f_dim3, 3)
        affine = backbone.theta

        # prediction, (batch, f_dim1, f_dim2, f_dim3)
        warping = layer.Warping(fixed_image_size=fixed_image_size)
        grid_fixed = tf.squeeze(warping.grid_ref, axis=0)  # (f_dim1, f_dim2, f_dim3, 3)
        pred_fixed_image = warping(inputs=[ddf, tf.squeeze(moving_image, axis=4)])
        pred_fixed_label = (
            warping(inputs=[ddf, moving_label]) if moving_label is not None else None
        )
        return affine, ddf, pred_fixed_image, pred_fixed_label, grid_fixed

Similar to `build_backbone` we also need to write the `build_affine_model` function,
which consists of building the model according to the networks' inputs, backbone and
loss function.

    def build_affine_model(
        moving_image_size: tuple,
        fixed_image_size: tuple,
        index_size: int,
        labeled: bool,
        batch_size: int,
        model_config: dict,
        loss_config: dict,
        ):
        """
        :param moving_image_size: (m_dim1, m_dim2, m_dim3)
        :param fixed_image_size: (f_dim1, f_dim2, f_dim3)
        :param index_size: int, the number of indices for identifying a sample
        :param labeled: bool, indicating if the data is labeled
        :param batch_size: int, size of mini-batch
        :param model_config: config for the model
        :param loss_config: config for the loss
        :return: the built tf.keras.Model
        """

        # inputs
        (moving_image, fixed_image, moving_label, fixed_label, indices) = build_inputs(
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            index_size=index_size,
            batch_size=batch_size,
            labeled=labeled,
        )

        # backbone
        backbone = build_backbone(
            image_size=fixed_image_size,
            out_channels=3,
            model_config=model_config,
            method_name=model_config["method"],
        )

        # forward
        affine, ddf, pred_fixed_image, pred_fixed_label, grid_fixed = affine_forward(
            backbone=backbone,
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_label=moving_label,
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
        )

        # build model
        inputs = {
            "moving_image": moving_image,
            "fixed_image": fixed_image,
            "indices": indices,
        }
        outputs = {"ddf": ddf, "affine": affine}
        model_name = model_config["method"].upper() + "RegistrationModel"
        if moving_label is None:  # unlabeled
            model = tf.keras.Model(
                inputs=inputs, outputs=outputs, name=model_name + "WithoutLabel"
            )
        else:  # labeled
            inputs["moving_label"] = moving_label
            inputs["fixed_label"] = fixed_label
            outputs["pred_fixed_label"] = pred_fixed_label
            model = tf.keras.Model(
                inputs=inputs, outputs=outputs, name=model_name + "WithLabel"
            )

        # add loss and metric
        model = add_ddf_loss(model=model, ddf=ddf, loss_config=loss_config)
        model = add_image_loss(
            model=model,
            fixed_image=fixed_image,
            pred_fixed_image=pred_fixed_image,
            loss_config=loss_config,
        )
        model = add_label_loss(
            model=model,
            grid_fixed=grid_fixed,
            fixed_label=fixed_label,
            pred_fixed_label=pred_fixed_label,
            loss_config=loss_config,
        )

        return model

Finally, the last step consists of adding the `build_affine_model` option to
`deepreg/model/network/build.py` to be able to parse it from the configuration file.

    def build_model(
        moving_image_size: tuple,
        fixed_image_size: tuple,
        index_size: int,
        labeled: bool,
        batch_size: int,
        model_config: dict,
        loss_config: dict,
    ):
        """
        Parsing algorithm types to model building functions
        :param moving_image_size: [m_dim1, m_dim2, m_dim3]
        :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
        :param index_size: dataset size
        :param labeled: true if the label of moving/fixed images are provided
        :param batch_size: mini-batch size
        :param model_config: model configuration, e.g. dictionary return from parser.yaml.load
        :param loss_config: loss configuration, e.g. dictionary return from parser.yaml.load
        :return: the built tf.keras.Model
        """
        if model_config["method"] in ["ddf", "dvf"]:
            return build_ddf_dvf_model(
                moving_image_size=moving_image_size,
                fixed_image_size=fixed_image_size,
                index_size=index_size,
                labeled=labeled,
                batch_size=batch_size,
                model_config=model_config,
                loss_config=loss_config,
            )
        elif model_config["method"] == "conditional":
            return build_conditional_model(
                moving_image_size=moving_image_size,
                fixed_image_size=fixed_image_size,
                index_size=index_size,
                labeled=labeled,
                batch_size=batch_size,
                model_config=model_config,
                loss_config=loss_config,
            )
        elif model_config["method"] == "affine":
            return build_affine_model(
                moving_image_size=moving_image_size,
                fixed_image_size=fixed_image_size,
                index_size=index_size,
                labeled=labeled,
                batch_size=batch_size,
                model_config=model_config,
                loss_config=loss_config,
            )
        else:
            raise ValueError("Unknown model method")

## Step 3: Testing (for contributing developers, optional for users)

Everyone is warmly welcome to make contributions to DeepReg and add corresponding unit
test for the newly added functions to `test/unit/`. Recommendations regarding testing
style can be found in the [contribution guidelines](../contributing/guide.html). Here is
a practical example of unit tests made for our affine model in
`test/unit/test_affine.py`:

    def test_affine_forward():
        """
        Testing that affine_forward function returns the tensors with correct shapes
        """

        moving_image_size = (1, 3, 5)
        fixed_image_size = (2, 4, 6)
        batch_size = 1

        global_net = build_backbone(
            image_size=fixed_image_size,
            out_channels=3,
            model_config={
                "backbone": "global",
                "global": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
            },
            method_name="affine",
        )

        # Check conditional mode network output shapes - Pass
        affine, ddf, pred_fixed_image, pred_fixed_label, grid_fixed = affine_forward(
            backbone=global_net,
            moving_image=tf.ones((batch_size,) + moving_image_size),
            fixed_image=tf.ones((batch_size,) + fixed_image_size),
            moving_label=tf.ones((batch_size,) + moving_image_size),
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
        )
        assert affine.shape == (batch_size,) + (4,) + (3,)
        assert ddf.shape == (batch_size,) + fixed_image_size + (3,)
        assert pred_fixed_image.shape == (batch_size,) + fixed_image_size
        assert pred_fixed_label.shape == (batch_size,) + fixed_image_size
        assert grid_fixed.shape == fixed_image_size + (3,)


    def test_build_affine_model():
        """
        Testing that build_affine_model function returns the tensors with correct shapes
        """
        moving_image_size = (1, 3, 5)
        fixed_image_size = (2, 4, 6)
        batch_size = 1

        model = build_affine_model(
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            index_size=1,
            labeled=True,
            batch_size=batch_size,
            model_config={
                "method": "affine",
                "backbone": "global",
                "global": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
            },
            loss_config={
                "dissimilarity": {
                    "image": {"name": "lncc", "weight": 0.1},
                    "label": {
                        "name": "multi_scale",
                        "weight": 1,
                        "multi_scale": {
                            "loss_type": "dice",
                            "loss_scales": [0, 1, 2, 4, 8, 16, 32],
                        },
                    },
                },
                "regularization": {"weight": 0.0, "energy_type": "bending"},
            },
        )

        inputs = {
            "moving_image": tf.ones((batch_size,) + moving_image_size),
            "fixed_image": tf.ones((batch_size,) + fixed_image_size),
            "indices": 1,
            "moving_label": tf.ones((batch_size,) + moving_image_size),
            "fixed_label": tf.ones((batch_size,) + fixed_image_size),
        }

        outputs = model(inputs)

        expected_outputs_keys = ["affine", "ddf", "pred_fixed_label"]
        assert all(keys in expected_outputs_keys for keys in outputs)
        assert outputs["pred_fixed_label"].shape == (batch_size,) + fixed_image_size
        assert outputs["affine"].shape == (batch_size,) + (4,) + (3,)
        assert outputs["ddf"].shape == (batch_size,) + fixed_image_size + (3,)

## Step 4: Set yaml configuration files

An example of yaml configuration file for the affine method is available in
`config/unpaired_labeled_affine.yaml`. For using both the GlobalNet backbone and affine
method you will need to add their aforementioned keyword "global" and "affine". Optional
parameters such as `out_kernel_initializer` or `num_channel_initial` can also be
specified. A snippet of `config/unpaired_labeled_affine.yaml` is shown below. Please see
the [configuration documentation](../docs/configuration.html) for more details.

    model:
    method: "affine"
    backbone:
      name: "global"
      out_kernel_initializer: "zeros"
      out_activation: ""
    global:
      num_channel_initial: 1
      extract_levels: [0, 1, 2, 3, 4]
