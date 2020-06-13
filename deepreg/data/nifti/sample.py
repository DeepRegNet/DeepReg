import random


def get_label_indices(num_labels, sample_label):
    if sample_label == "sample":  # sample a random label
        return [random.randrange(num_labels)]
    elif sample_label == "first":  # use the first label
        return [0]
    elif sample_label == "all":  # use all labels
        return list(range(num_labels))
    else:
        raise ValueError("Unknown label sampling policy %s" % sample_label)
