import subprocess

import pytest


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


class TestTutorial:
    @pytest.mark.parametrize(
        "name",
        [
            "custom_backbone",
            "custom_image_label_loss",
            "custom_parameterized_image_label_loss",
        ],
    )
    def test_registry(self, name: str):
        cmds = [f"python examples/{name}.py"]
        execute_commands(cmds)
