# download the demo data

import os

project_dir = r"demos/unpaired_us_prostate_cv"
os.chdir(project_dir)

DATA_PATH = "./data"

temp_file = os.path.join(DATA_PATH, "master.zip")
origin = "https://github.com/ucl-candi/dataset_trus3d/archive/master.zip"

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

os.system("wget -P %s  %s" % (DATA_PATH, origin))
os.system("unzip %s -d %s" % (temp_file, DATA_PATH))
os.remove(temp_file)

print(
    "TRUS 3d data downloaded: %s."
    % os.path.abspath(os.path.join(DATA_PATH, "dataset_trus3d-master"))
)
