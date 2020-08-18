import os
import shutil
import zipfile

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import get_file

PROJECT_DIR = r"demos/unpaired_ct_abdomen"
os.chdir(PROJECT_DIR)

ORIGIN = "https://github.com/ucl-candi/datasets_deepreg_demo/archive/abdct.zip"
ZIP_PATH = "abdct.zip"
DATA_PATH = "dataset"

get_file(os.path.abspath(ZIP_PATH), ORIGIN)

zf = zipfile.ZipFile(ZIP_PATH)
filenames_all = [fn for fn in zf.namelist() if fn.split(".")[-1] == "gz"]
num_data = int(len(filenames_all) / 2)
# check indices
filenames_indices = list(
    set([int(fn.split("/")[-1].split(".")[0]) for fn in filenames_all])
)
if len(filenames_indices) is not num_data:
    raise ("Images and labels are not in pairs.")

print("\nAbdominal CT data downloaded with %d image-label pairs.", num_data)

ratio_val = 0.1
ratio_test = 0.15
num_val = int(num_data * ratio_val)
num_test = int(num_data * ratio_test)
num_train = num_data - num_val - num_test

print(
    "Extracting data into %d-%d-%d for train-val-test (%0.2f-%0.2f-%0.2f)..."
    % (num_train, num_val, num_test, 1 - ratio_val - ratio_test, ratio_val, ratio_test)
)

# extract to respective folders
folders = [os.path.join(DATA_PATH, dn) for dn in ["train", "val", "test"]]
if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.mkdir(DATA_PATH)
for fn in folders:
    os.mkdir(fn)
    os.mkdir(os.path.join(fn, "images"))
    os.mkdir(os.path.join(fn, "labels"))

for filename in filenames_all:
    # images or labels
    if filename.startswith("datasets_deepreg_demo-abdct/dataset/images"):
        typename = "images"
    elif filename.startswith("datasets_deepreg_demo-abdct/dataset/labels"):
        typename = "labels"
    else:
        continue
    # train, val or test
    idx = filenames_indices.index(int(filename.split("/")[-1].split(".")[0]))
    if idx < num_train:  # train
        fidx = 0
    elif idx < (num_train + num_val):  # val
        fidx = 1
    else:  # test
        fidx = 2
    filename_dst = os.path.join(folders[fidx], typename, filename.split("/")[-1])
    with zf.open(filename) as sf, open(filename_dst, "wb") as df:
        shutil.copyfileobj(sf, df)
    # re-encode the label files - hard-coded using 13 of them regardless exists or not
    if typename == "labels":
        img = nib.load(filename_dst)
        img1 = np.stack(
            [np.asarray(img.dataobj) == label_id for label_id in range(1, 14)], axis=3
        )
        img1 = nib.Nifti1Image(img1.astype(np.int8), img.affine)
        img1.to_filename(filename_dst)

os.remove(ZIP_PATH)

print("Done. \n")
