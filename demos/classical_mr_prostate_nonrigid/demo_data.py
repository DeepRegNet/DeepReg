"""
Download the demo data
"""
import os

from tensorflow.keras.utils import get_file

MAIN_PATH = os.getcwd()
PROJECT_DIR = r"demos/classical_mr_prostate_nonrigid"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
FILE_PATH = os.path.abspath(os.path.join(DATA_PATH, "demo2.h5"))
ORIGIN = "https://github.com/YipengHu/example-data/raw/master/promise12/demo2.h5"

if os.path.exists(DATA_PATH):
    os.remove(FILE_PATH)
else:
    os.mkdir(DATA_PATH)
get_file(FILE_PATH, ORIGIN)
print("Prostate MR data downloaded: %s." % FILE_PATH)

os.chdir(MAIN_PATH)
