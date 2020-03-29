import deepreg.data.augmentation as aug


def preprocess(dataset,
               moving_image_shape, fixed_image_shape,
               training,
               shuffle_buffer_size, repeat: bool, batch_size):
    """
    shuffle, repeat, batch, augmentation
    :param dataset:
    :param moving_image_shape:
    :param fixed_image_shape:
    :param training:
    :param batch_size:
    :param repeat:
    :param shuffle_buffer_size:
    :return:
    """
    if training and shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=training)
    if training:
        # TODO add cropping, but crop first or rotation first?
        affine_transform = aug.AffineTransformation3D(moving_image_size=moving_image_shape,
                                                      fixed_image_size=fixed_image_shape,
                                                      batch_size=batch_size)
        dataset = dataset.map(affine_transform.transform)
    return dataset
