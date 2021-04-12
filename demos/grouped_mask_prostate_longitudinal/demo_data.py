"""
Download the demo data and sort them into train, val and test in h5 files
"""
import os
import shutil
import zipfile

import h5py
from scipy import ndimage
from tensorflow.keras.utils import get_file

PROJECT_DIR = "demos/grouped_mask_prostate_longitudinal"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
ZIP_FILE = "data"
ORIGIN = "https://github.com/YipengHu/example-data/raw/master/longi-masks/data.zip"

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.mkdir(DATA_PATH)

zip_file = os.path.join(DATA_PATH, ZIP_FILE + ".zip")
get_file(os.path.abspath(zip_file), ORIGIN)
with zipfile.ZipFile(zip_file, "r") as zf:
    zf.extractall(DATA_PATH)
os.remove(zip_file)

print("\nMask data downloaded: %s." % os.path.abspath(DATA_PATH))

## now read the data and convert to train/val/test
ratio_val = 0.1
ratio_test = 0.2

data_filename = os.path.join(DATA_PATH, ZIP_FILE + ".h5")
fid_data = h5py.File(data_filename, "r")
num_data = len(fid_data)
ids_group, ids_ob = [], []
for f in fid_data:
    ds, ig, io = fid_data[f].name.split("-")
    if ds == "/group":
        ids_group.append(int(ig))
        ids_ob.append(int(io))
ids_group_unique = list(set(ids_group))
num_group = len(ids_group_unique)
num_val = int(num_group * ratio_val)
num_test = int(num_group * ratio_test)
num_train = num_group - num_val - num_test

print("Found %d data in %d groups." % (num_data, num_group))
print(
    "Dividing into %d-%d-%d for train-val-test (%0.2f-%0.2f-%0.2f)..."
    % (
        num_train,
        num_val,
        num_test,
        1 - ratio_val - ratio_test,
        ratio_val,
        ratio_test,
    )
)

# write
fid_image, fid_label = [], []
folders = [
    os.path.join(DATA_PATH, "train"),
    os.path.join(DATA_PATH, "val"),
    os.path.join(DATA_PATH, "test"),
]
for fn in folders:
    os.mkdir(fn)
    fid_label.append(h5py.File(os.path.join(fn, "labels.h5"), "w"))
    fid_image.append(h5py.File(os.path.join(fn, "images.h5"), "w"))

for i in range(num_data):
    dataset_name = "group-%d-%d" % (ids_group[i], ids_ob[i])
    pos_group = ids_group_unique.index(ids_group[i])
    if pos_group < num_train:  # train
        idf = 0
    elif pos_group < (num_train + num_val):  # val
        idf = 1
    else:  # test
        idf = 2
    data = fid_data[dataset_name]
    fid_label[idf].create_dataset(
        dataset_name, shape=data.shape, dtype=data.dtype, data=data
    )
    fid_label[idf].flush()
    image = ndimage.gaussian_filter(
        data, sigma=3, output="float32"
    )  # smoothing with gaussian
    fid_image[idf].create_dataset(
        dataset_name, shape=image.shape, dtype=image.dtype, data=image
    )
    fid_image[idf].flush()
    # print(idf,dataset_name)

# close all
fid_data.close()
for idf in range(len(folders)):
    fid_label[idf].close()
    fid_image[idf].close()
os.remove(data_filename)

print("Done. \n")

## now download the pretrained model
MODEL_PATH = os.path.join(DATA_PATH, "pretrained")
if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)
os.mkdir(MODEL_PATH)

ZIP_PATH = "grouped_mask_prostate_longitudinal_1"
ORIGIN = "https://github.com/DeepRegNet/deepreg-model-zoo/raw/master/demo/grouped_mask_prostate_longitudinal/20210110.zip"

zip_file = os.path.join(MODEL_PATH, ZIP_PATH + ".zip")
get_file(os.path.abspath(zip_file), ORIGIN)
with zipfile.ZipFile(zip_file, "r") as zf:
    zf.extractall(path=MODEL_PATH)
os.remove(zip_file)

print(
    "pretrained model is downloaded and unzipped in %s." % os.path.abspath(MODEL_PATH)
)
