"""
Download the demo data
"""
import os
import shutil
import zipfile

from tensorflow.keras.utils import get_file

PROJECT_DIR = "demos/unpaired_us_prostate_cv"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
DATA_REPO = "dataset_trus3d-master"
ZIP_PATH = "master.zip"
ORIGIN = "https://github.com/ucl-candi/dataset_trus3d/archive/master.zip"

get_file(os.path.abspath(ZIP_PATH), ORIGIN)
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall()

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
shutil.move(DATA_REPO, DATA_PATH)
os.remove(ZIP_PATH)

print("TRUS 3d data downloaded: %s." % os.path.abspath(DATA_PATH))

# Download the pretrained models
MODEL_PATH = os.path.join(DATA_PATH, "pretrained")
if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)
os.mkdir(MODEL_PATH)

ZIP_PATH = "unpaired_us_prostate_cv_1"
ORIGIN = "https://github.com/DeepRegNet/deepreg-model-zoo/raw/master/demo/unpaired_us_prostate_cv/20210110.zip"

zip_file = os.path.join(MODEL_PATH, ZIP_PATH + ".zip")
get_file(os.path.abspath(zip_file), ORIGIN)
with zipfile.ZipFile(zip_file, "r") as zf:
    zf.extractall(path=MODEL_PATH)
os.remove(zip_file)

print(
    "pretrained model is downloaded and unzipped in %s." % os.path.abspath(MODEL_PATH)
)
