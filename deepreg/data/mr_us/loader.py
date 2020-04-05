from deepreg.data.loader import GeneratorDataLoader


class DataLoader(GeneratorDataLoader):
    def __init__(self):
        super(DataLoader, self).__init__()

    def split_indices(self, indices: list):
        image_index, label_index = indices
        return image_index, label_index

    def image_index_to_dir(self, image_index):
        return "image{image_index:d}".format(image_index=image_index)
