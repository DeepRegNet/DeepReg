from deepreg.data.loader import GeneratorDataLoader


class DataLoader(GeneratorDataLoader):
    def __init__(self):
        super(DataLoader, self).__init__()

    def split_indices(self, indices: list):
        pid1, vid1, pid2, vid2, label_index = indices
        image_index = (pid1, vid1, pid2, vid2)
        return image_index, label_index

    def image_index_to_dir(self, image_index):
        pid1, vid1, pid2, vid2 = image_index
        return "pid1_{pid1:d}_vid1_{vid1:d}_pid2_{pid2:d}_vid2_{vid2:d}".format(pid1=pid1, vid1=vid1, pid2=pid2,
                                                                                vid2=vid2)
