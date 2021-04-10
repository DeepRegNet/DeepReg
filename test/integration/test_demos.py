import os
import shutil
import subprocess

import pytest

from deepreg import log

logger = log.get(__name__)


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


def check_vis_single_config_demo(name):
    time_stamp = sorted(os.listdir(f"demos/{name}/logs_predict"))[0]
    pair_number = sorted(os.listdir(f"demos/{name}/logs_predict/{time_stamp}/test"))[-1]
    cmd = [
        f"deepreg_vis -m 2 -i 'demos/{name}/logs_predict/{time_stamp}/test/{pair_number}/moving_image.nii.gz, demos/{name}/logs_predict/{time_stamp}/test/{pair_number}/pred_fixed_image.nii.gz, demos/{name}/logs_predict/{time_stamp}/test/{pair_number}/fixed_image.nii.gz' --slice-inds '0,1,2' -s demos/{name}/logs_predict"
    ]
    execute_commands([cmd])
    assert os.path.exists(f"demos/{name}/logs_predict/visualisation.png")


def check_vis_unpaired_ct_abdomen(name, method):
    time_stamp = sorted(os.listdir(f"demos/{name}/logs_predict/{method}"))[0]
    pair_number = sorted(
        os.listdir(f"demos/{name}/logs_predict/{method}/{time_stamp}/test")
    )[-1]
    cmd = [
        f"deepreg_vis -m 2 -i 'demos/{name}/logs_predict/{method}/{time_stamp}/test/{pair_number}/moving_image.nii.gz, demos/{name}/logs_predict/{method}/{time_stamp}/test/{pair_number}/pred_fixed_image.nii.gz, demos/{name}/logs_predict/{method}/{time_stamp}/test/{pair_number}/fixed_image.nii.gz' --slice-inds '0,1,2' -s demos/{name}/logs_predict"
    ]
    execute_commands([cmd])
    assert os.path.exists(f"demos/{name}/logs_predict/visualisation.png")


def check_vis_classical_demo(name):
    cmd = [
        f"deepreg_vis -m 2 -i 'demos/{name}/logs_reg/moving_image.nii.gz, demos/{name}/logs_reg/warped_moving_image.nii.gz, demos/{name}/logs_reg/fixed_image.nii.gz' --slice-inds '0,1,2' -s demos/{name}/logs_reg"
    ]
    execute_commands([cmd])
    assert os.path.exists(f"demos/{name}/logs_reg/visualisation.png")


def execute_commands(cmds):
    for cmd in cmds:
        try:
            logger.info(f"Running {cmd}")
            out = subprocess.check_output(cmd, shell=True).decode("utf-8")
            logger.info(out)
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
        cmds = [f"python demos/{name}/demo_{x}.py --test" for x in ["train", "predict"]]

        execute_commands(cmds)
        check_vis_single_config_demo(name)

    @pytest.mark.parametrize(
        "method",
        ["comb", "unsup", "weakly"],
    )
    def test_unpaired_ct_abdomen(self, method):
        """this demo has multiple configuration file"""
        name = "unpaired_ct_abdomen"
        remove_files(name)

        # execute data
        cmds = [f"python demos/{name}/demo_data.py"]
        execute_commands(cmds)

        # check temporary files are removed
        check_files(name)

        # execute train, predict sequentially
        cmds = [
            f"python demos/{name}/demo_{x}.py --method {method} --test"
            for x in ["train", "predict"]
        ]

        execute_commands(cmds)
        check_vis_unpaired_ct_abdomen(name, method)

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
        cmds = [f"python demos/{name}/demo_register.py --test"]

        execute_commands(cmds)
        check_vis_classical_demo(name)
