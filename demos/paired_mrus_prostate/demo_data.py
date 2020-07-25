"""
Download the demo data
"""
import os
import shutil
import zipfile

from tensorflow.keras.utils import get_file

PROJECT_DIR = r"demos/paired_mrus_prostate"
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

print("MR and ultrasound data downloaded: %s." % os.path.abspath(DATA_PATH))
