import os
import shutil
import subprocess

import pytest


def remove_files(name):
    dir_name = os.path.join("demos", name)

    # remove zip files
    files = os.listdir(dir_name)
    for file in files:
        if file.endswith(".zip"):
            os.remove(os.path.join(dir_name, file))

    # remove output folders
    paths = [
        os.path.join(dir_name, x)
        for x in ["dataset", "logs_train", "logs_predict", "logs_reg"]
    ]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)


def check_files(name):
    """make sure dataset folder exist but there is no zip files"""
    dir_name = os.path.join("demos", name)

    # assert dataset folder exists
    assert os.path.exists(os.path.join(dir_name, "dataset"))

    # assert no zip files
    files = os.listdir(dir_name)
    files = [x for x in files if x.endswith(".zip")]
    assert len(files) == 0


def execute_commands(cmds):
    for cmd in cmds:
        try:
            print(f"Running {cmd}")
            out = subprocess.check_output(cmd, shell=True).decode("utf-8")
            print(out)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Command {cmd} return with err {e.returncode} {e.output}"
            )


class TestDemo:
    @pytest.mark.parametrize(
        "name",
        [
            "grouped_mask_prostate_longitudinal",
            "grouped_mr_heart",
            "paired_ct_lung",
            "paired_mrus_brain",
            "paired_mrus_prostate",
            "unpaired_ct_lung",
            "unpaired_mr_brain",
            "unpaired_us_prostate_cv",
        ],
        ids=[
            "grouped_mask_prostate_longitudinal",
            "grouped_mr_heart",
            "paired_ct_lung",
            "paired_mrus_brain",
            "paired_mrus_prostate",
            "unpaired_ct_lung",
            "unpaired_mr_brain",
            "unpaired_us_prostate_cv",
        ],
    )
    def test_single_config_demo(self, name):
        """each demo has one single configuration file"""
        remove_files(name)

        # execute data
        cmds = [f"python demos/{name}/demo_data.py"]
        execute_commands(cmds)

        # check temporary files are removed
        check_files(name)

        # execute train, predict sequentially
        cmds = [f"python demos/{name}/demo_{x}.py" for x in ["train", "predict"]]

        execute_commands(cmds)

    def test_unpaired_ct_abdomen(self):
        """this demo has multiple configuration file"""
        name = "unpaired_ct_abdomen"
        remove_files(name)

        # execute data
        cmds = [f"python demos/{name}/demo_data.py"]
        execute_commands(cmds)

        # check temporary files are removed
        check_files(name)

        # execute train, predict sequentially
        cmds = []
        for method in ["comb", "unsup", "weakly"]:
            cmds += [
                f"python demos/{name}/demo_{x}.py --method {method}"
                for x in ["train", "predict"]
            ]

        execute_commands(cmds)

    @pytest.mark.parametrize(
        "name",
        [
            "classical_ct_headneck_affine",
            "classical_mr_prostate_nonrigid",
        ],
    )
    def test_classical_demo(self, name):
        remove_files(name)

        # execute data
        cmds = [f"python demos/{name}/demo_data.py"]
        execute_commands(cmds)

        # check temporary files are removed
        check_files(name)

        # execute data, register
        cmds = [f"python demos/{name}/demo_register.py"]

        execute_commands(cmds)
