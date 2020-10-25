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
        os.path.join(dir_name, x) for x in ["dataset", "logs_train", "logs_predict"]
    ]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)


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
        ],
    )
    def test_simple_demo(self, name):
        """each demo has one single configuration file"""
        remove_files(name)

        # execute data, train, predict sequentially
        cmds = [
            f"python demos/{name}/demo_{x}.py" for x in ["data", "train", "predict"]
        ]

        execute_commands(cmds)

    def test_unpaired_ct_abdomen(self):
        """this demo has multiple configuration file"""
        name = "unpaired_ct_abdomen"
        remove_files(name)

        # execute data, train, predict sequentially
        cmds = [f"python demos/{name}/demo_data.py"]
        for method in ["comb", "unsup", "weakly"]:
            cmds += [
                f"python demos/{name}/demo_{x}.py --method {method}"
                for x in ["train", "predict"]
            ]

        execute_commands(cmds)
