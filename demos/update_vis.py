import os
import subprocess


def check_vis_single_config_demo(name, slice):
    time_stamp = sorted(os.listdir(f"demos/{name}/logs_predict"))[-1]
    pair_number = sorted(os.listdir(f"demos/{name}/logs_predict/{time_stamp}/test"))[-1]
    cmd = [
        f"deepreg_vis -m 2 -i 'demos/{name}/logs_predict/{time_stamp}/test/{pair_number}/moving_image.nii.gz, demos/{name}/logs_predict/{time_stamp}/test/{pair_number}/pred_fixed_image.nii.gz, demos/{name}/logs_predict/{time_stamp}/test/{pair_number}/fixed_image.nii.gz' --slice-inds {slice} -s /home/zcemsus/deepreg_demo_vis/all_assets --fname {name}.png"
    ]
    execute_commands([cmd])


def check_vis_unpaired_ct_abdomen(name, method, slice):
    time_stamp = sorted(os.listdir(f"demos/{name}/logs_predict/{method}"))[-1]
    pair_number = sorted(
        os.listdir(f"demos/{name}/logs_predict/{method}/{time_stamp}/test")
    )[-1]
    cmd = [
        f"deepreg_vis -m 2 -i 'demos/{name}/logs_predict/{method}/{time_stamp}/test/{pair_number}/moving_image.nii.gz, demos/{name}/logs_predict/{method}/{time_stamp}/test/{pair_number}/pred_fixed_image.nii.gz, demos/{name}/logs_predict/{method}/{time_stamp}/test/{pair_number}/fixed_image.nii.gz' --slice-inds {slice} -s /home/zcemsus/deepreg_demo_vis/all_assets --fname {name}.png"
    ]
    execute_commands([cmd])


def check_vis_classical_demo(name, slice):
    cmd = [
        f"deepreg_vis -m 2 -i 'demos/{name}/logs_reg/moving_image.nii.gz, demos/{name}/logs_reg/warped_moving_image.nii.gz, demos/{name}/logs_reg/fixed_image.nii.gz' --slice-inds {slice} -s /home/zcemsus/deepreg_demo_vis/all_assets --fname {name}.png"
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

single_config_names = ["grouped_mask_prostate_longitudinal", "grouped_mr_heart","paired_ct_lung", "paired_mrus_brain", "paired_mrus_prostate", "unpaired_ct_lung", "unpaired_mr_brain", "unpaired_us_prostate_cv"]
slices = ['10,16,20', '14,10,20', '64,50,72', '190,128,96', '12,20,36', '40,48,56', '20,32,44', '50,65,35']
for name, slice in zip(single_config_names, slices):
    execute_commands([f'python demos/{name}/demo_data.py'])
    execute_commands([f'python demos/{name}/demo_predict.py'])
    check_vis_single_config_demo(name, slice)

for name, slice in zip(["unpaired_ct_abdomen"], ['35,50,65']):
    execute_commands([f'python demos/{name}/demo_data.py'])
    execute_commands([f'python demos/unpaired_ct_abdomen/demo_predict.py --method comb'])
    check_vis_unpaired_ct_abdomen(name, "comb", slice)

for name, slice in zip(["classical_ct_headneck_affine", "classical_mr_prostate_nonrigid"], ['4,8,12', '4,8,12']):
    execute_commands([f'python demos/{name}/demo_data.py'])
    execute_commands([f'python demos/{name}/demo_register.py'])
    check_vis_classical_demo(name, slice)




