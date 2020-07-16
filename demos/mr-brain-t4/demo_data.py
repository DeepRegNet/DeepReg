from os import listdir
from os.path import join

import nibabel as nib
import numpy as np

DATA_DIR = "/home/acasamitjana/Data/Learn2Reg"
data_splits = ["train", "valid"]
num_labels = 2
# Download data

# Extract data and rename directories

# Normalize labels
for ds in data_splits:
    path = join(DATA_DIR, ds, "labels")
    files = listdir(path)
    for f in files:
        proxy = nib.load(join(path, f))
        data = np.asarray(proxy.dataobj)
        data = np.double(data > 0.5)
        img = nib.Nifti1Image(data, proxy.affine)
        nib.save(img, join(path, f))
