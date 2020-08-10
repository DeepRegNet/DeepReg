"""
Download the demo data
"""
import os
import shutil
import zipfile

# import h5py
from tensorflow.keras.utils import get_file

PROJECT_DIR = r"demos/grouped_mask_prostate_longitudinal"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.mkdir(DATA_PATH)

ZIP_FILE = "data"
ORIGIN = "https://github.com/YipengHu/example-data/raw/master/longi-masks/data.zip"

zip_file = os.path.join(DATA_PATH, ZIP_FILE + ".zip")
get_file(os.path.abspath(zip_file), ORIGIN)
with zipfile.ZipFile(zip_file, "r") as zf:
    zf.extractall(DATA_PATH)
os.remove(zip_file)

print("\nMask data downloaded: %s." % os.path.abspath(DATA_PATH))


## now read the data and save them into

"""
## now download the pre-trained model
MODEL_PATH = os.path.join(DATA_PATH, "pre-trained")
if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)
os.mkdir(MODEL_PATH)

ZIP_PATH = "paired_mrus_prostate-ckpt"
ORIGIN = "https://github.com/DeepRegNet/deepreg-model-zoo/raw/master/paired_mrus_prostate-ckpt.zip"

zip_file = os.path.join(MODEL_PATH, ZIP_PATH + ".zip")
get_file(os.path.abspath(zip_file), ORIGIN)
with zipfile.ZipFile(zip_file, "r") as zf:
    zf.extractall(path=MODEL_PATH)
os.remove(zip_file)

print(
    "Pre-trained model is downloaded and unzipped in %s." % os.path.abspath(MODEL_PATH)
)
"""
