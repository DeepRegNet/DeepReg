import argparse
import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def check_vis_single_config_demo(name, slice):
    time = sorted(os.listdir(f"demos/{name}/logs_predict"))[-1]
    pair = sorted(os.listdir(f"demos/{name}/logs_predict/{time}/test"))[-1]
    cmd = [
        f"deepreg_vis -m 2 -i "
        f"'demos/{name}/logs_predict/{time}/test/{pair}/moving_image.nii.gz,"
        f"demos/{name}/logs_predict/{time}/test/{pair}/pred_fixed_image.nii.gz,"
        f"demos/{name}/logs_predict/{time}/test/{pair}/fixed_image.nii.gz'"
        f" --slice-inds {slice} -s /home/zcemsus/deepreg_demo_vis/all_assets"
        f" --fname {name}.png"
    ]
    execute_commands([cmd])


def check_vis_unpaired_ct_abdomen(name, method, slice):
    time = sorted(os.listdir(f"demos/{name}/logs_predict/{method}"))[-1]
    pair = sorted(os.listdir(f"demos/{name}/logs_predict/{method}/{time}/test"))[-1]
    cmd = [
        f"deepreg_vis -m 2 -i "
        f"'demos/{name}/logs_predict/{method}/{time}/test/{pair}/moving_image.nii.gz,"
        f"demos/{name}/logs_predict/{method}/{time}/test/{pair}/"
        f"pred_fixed_image.nii.gz,"
        f"demos/{name}/logs_predict/{method}/{time}/test/{pair}/fixed_image.nii.gz'"
        f" --slice-inds {slice} -s /home/zcemsus/deepreg_demo_vis/all_assets"
        f" --fname {name}.png"
    ]
    execute_commands([cmd])


def check_vis_classical_demo(name, slice):
    cmd = [
        f"deepreg_vis -m 2 -i "
        f"'demos/{name}/logs_reg/moving_image.nii.gz,"
        f"demos/{name}/logs_reg/warped_moving_image.nii.gz,"
        f"demos/{name}/logs_reg/fixed_image.nii.gz'"
        f" --slice-inds {slice} -s /home/zcemsus/deepreg_demo_vis/all_assets"
        f" --fname {name}.png"
    ]
    execute_commands([cmd])


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


def main(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--demos",
        "-d",
        help="Comma separated string of demo names for which to update visualisation"
        "e.g. 'paired_ct_lung, classical_ct_headneck_affine, unpaired_ct_abdomen'"
        "if not passed then all visualisations are updated",
        type=str,
        default=None,
    )

    single_config_names = [
        "grouped_mask_prostate_longitudinal",
        "grouped_mr_heart",
        "paired_ct_lung",
        "paired_mrus_brain",
        "paired_mrus_prostate",
        "unpaired_ct_lung",
        "unpaired_mr_brain",
        "unpaired_us_prostate_cv",
    ]
    single_config_slices = [
        "10,16,20",
        "14,10,20",
        "64,50,72",
        "190,128,96",
        "12,20,36",
        "40,48,56",
        "20,32,44",
        "50,65,35",
    ]

    multi_config_names = ["unpaired_ct_abdomen"]
    multi_config_slices = ["35,50,65"]
    multi_config_methods = ["comb"]

    classical_names = ["classical_ct_headneck_affine", "classical_mr_prostate_nonrigid"]
    classical_slices = ["4,8,12", "4,8,12"]

    demo_names = [elem.strip() for elem in args.demos.split(",")]

    for demo_name in demo_names:

        if demo_name in single_config_names:
            ind = single_config_names.index(demo_name)
            execute_commands([f"python demos/{demo_name}/demo_data.py"])
            execute_commands([f"python demos/{demo_name}/demo_predict.py"])
            check_vis_single_config_demo(demo_name, single_config_slices[ind])

        elif demo_name in multi_config_names:
            ind = multi_config_names.index(demo_name)
            execute_commands([f"python demos/{demo_name}/demo_data.py"])
            execute_commands(
                [
                    f"python demos/{demo_name}/demo_predict.py --method {multi_config_methods[ind]}"
                ]
            )
            check_vis_unpaired_ct_abdomen(
                demo_name, multi_config_methods[ind], multi_config_slices[ind]
            )

        elif demo_name in classical_names:
            ind = classical_names.index(demo_name)
            execute_commands([f"python demos/{demo_name}/demo_data.py"])
            execute_commands([f"python demos/{demo_name}/demo_register.py"])
            check_vis_classical_demo(demo_name, classical_slices[ind])

        else:
            raise Exception(f"{demo_name} was not recognised as a demo name")

    demo_names_updated_string = ", ".join(demo_names)
    print(f"Updated: {demo_names_updated_string}")


if __name__ == "__main__":
    main()
