import os
import shutil
import zipfile

import nibabel as nib
from scipy import ndimage
from tensorflow.keras.utils import get_file

output_pixdim = 1.5

PROJECT_DIR = r"demos/grouped_mr_heart"
os.chdir(PROJECT_DIR)

ORIGIN = "https://github.com/ucl-candi/datasets_deepreg_demo/archive/myops.zip"
ZIP_PATH = "myops.zip"
DATA_PATH = "dataset"

get_file(os.path.abspath(ZIP_PATH), ORIGIN)

zf = zipfile.ZipFile(ZIP_PATH)
filenames_all = [fn for fn in zf.namelist() if fn.split(".")[-1] == "gz"]
num_data = int(len(filenames_all) / 3)
# check indices
filenames_indices = list(
    set([int(fn.split("/")[-1].split("_")[0]) for fn in filenames_all])
)
if len(filenames_indices) is not num_data:
    raise ("Missing data in image groups.")

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.mkdir(DATA_PATH)

print(
    "\nCMR data from %d subjects downloaded, being extracted and resampled..."
    % num_data
)
print("This may take a few minutes...")

# extract into image groups
images_path = os.path.join(DATA_PATH, "images")
os.mkdir(images_path)

for filename in filenames_all:
    # groups, here same as subjects
    idx, seq_name = filename.split("/")[-1].split("_")
    idx_group = filenames_indices.index(int(idx))
    group_path = os.path.join(images_path, "subject" + "%03d" % idx_group)
    if os.path.exists(group_path) is not True:
        os.mkdir(group_path)

    # extract image
    img_path = os.path.join(group_path, seq_name)
    with zf.open(filename) as sf, open(img_path, "wb") as df:
        shutil.copyfileobj(sf, df)
    # pre-processing
    img = nib.load(img_path)
    img = nib.Nifti1Image(
        ndimage.zoom(
            img.dataobj, [pd / output_pixdim for pd in img.header.get_zooms()]
        ),
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    )  # to a generic affine after resampling
    img.to_filename(img_path)

os.remove(ZIP_PATH)

print("Done")

ratio_val = 0.05
ratio_test = 0.10
num_val = int(num_data * ratio_val)
num_test = int(num_data * ratio_test)
num_train = num_data - num_val - num_test

print(
    "Splitting data into %d-%d-%d for train-val-test (%0.2f-%0.2f-%0.2f)..."
    % (num_train, num_val, num_test, 1 - ratio_val - ratio_test, ratio_val, ratio_test)
)

# move images to respective folders
folders = [os.path.join(DATA_PATH, dn) for dn in ["train", "val", "test"]]

for fn in folders:
    os.mkdir(fn)
    os.mkdir(os.path.join(fn, "images"))

group_names = os.listdir(images_path)
for group in group_names:
    idx = group_names.index(group)
    if idx < num_train:  # train
        fidx = 0
    elif idx < (num_train + num_val):  # val
        fidx = 1
    else:  # test
        fidx = 2
    shutil.move(os.path.join(images_path, group), os.path.join(folders[fidx], "images"))

os.rmdir(images_path)

print("Done. \n")

# Download the pre-trained models
MODEL_PATH = os.path.join(DATA_PATH, "pre-trained")
if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)
os.mkdir(MODEL_PATH)

ZIP_PATH = "grouped_mr_heart-ckpt"
ORIGIN = "https://github.com/DeepRegNet/deepreg-model-zoo/raw/master/grouped_mr_heart-ckpt.zip"

zip_file = os.path.join(MODEL_PATH, ZIP_PATH + ".zip")
get_file(os.path.abspath(zip_file), ORIGIN)
with zipfile.ZipFile(zip_file, "r") as zf:
    zf.extractall(path=MODEL_PATH)
os.remove(zip_file)

print(
    "Pre-trained model is downloaded and unzipped in %s." % os.path.abspath(MODEL_PATH)
)
