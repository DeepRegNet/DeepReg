"""
Download the demo data
"""
import os
import shutil
import zipfile

from tensorflow.keras.utils import get_file

PROJECT_DIR = "demos/paired_mrus_prostate"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
ZIP_PATH = "example-data-mrus"
ORIGIN = "https://github.com/yipenghu/example-data/archive/mrus.zip"

zip_file = ZIP_PATH + ".zip"
get_file(os.path.abspath(zip_file), ORIGIN)
with zipfile.ZipFile(zip_file, "r") as zf:
    zf.extractall()

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.rename(ZIP_PATH, DATA_PATH)
os.remove(zip_file)

print("\nMR and ultrasound data downloaded: %s." % os.path.abspath(DATA_PATH))

# now split the data in to num_part partitions
num_part = 11

data_types = ["moving_images", "moving_labels", "fixed_images", "fixed_labels"]
filenames = [sorted(os.listdir(os.path.join(DATA_PATH, fn))) for fn in data_types]
num_files = [len(x) for x in filenames]
if len(set(num_files)) != 1:
    raise ValueError(
        "Number of data are not the same between moving/fixed/images/labels. "
        "Please run this download script again."
    )
num_data = num_files[0]

for idx in range(num_part):  # create partition folders
    os.makedirs(os.path.join(DATA_PATH, "part%02d" % idx))
    for fn in data_types:
        os.makedirs(os.path.join(DATA_PATH, "part%02d" % idx, fn))

for idx in range(num_data):  # copy all files to part folders
    for ifn in range(len(data_types)):
        os.rename(
            os.path.join(DATA_PATH, data_types[ifn], filenames[ifn][idx]),
            os.path.join(
                DATA_PATH,
                "part%02d" % (idx % num_part),
                data_types[ifn],
                filenames[ifn][idx],
            ),
        )

for fn in data_types:  # remove the old type folders
    shutil.rmtree(os.path.join(DATA_PATH, fn))

print("All data are partitioned into %d folders." % num_part)

## now download the pre-trained model
MODEL_PATH = os.path.join(DATA_PATH, "pretrained")
if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)
os.mkdir(MODEL_PATH)

ZIP_PATH = "checkpoint"
ORIGIN = "https://github.com/DeepRegNet/deepreg-model-zoo/raw/master/demo/paired_mrus_prostate/20210110.zip"

zip_file = os.path.join(MODEL_PATH, ZIP_PATH + ".zip")
get_file(os.path.abspath(zip_file), ORIGIN)
with zipfile.ZipFile(zip_file, "r") as zf:
    zf.extractall(path=MODEL_PATH)
os.remove(zip_file)

print(
    "Pre-trained model is downloaded and unzipped in %s." % os.path.abspath(MODEL_PATH)
)
