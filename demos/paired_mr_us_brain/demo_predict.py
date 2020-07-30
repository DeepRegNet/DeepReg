import os

import matplotlib.pyplot as plt

from deepreg.predict import predict

######## PREDICTION ########


log_dir = "learn2reg_t1_paired_train_logs"
ckpt_path = os.path.join("logs", log_dir, "save", "weights-epoch2.ckpt")
config_path = "logs/learn2reg_t1_paired_train_logs/config.yaml"

gpu = ""
gpu_allow_growth = False
predict(
    gpu=gpu,
    gpu_allow_growth=gpu_allow_growth,
    config_path=config_path,
    ckpt_path=ckpt_path,
    mode="test",
    batch_size=1,
    log_dir=log_dir,
    sample_label="all",
)

# the numerical metrics are saved in the logs directory specified

######## VISUALISATION ########

# Now lets load in a few samples from the predicitons and plot them

# change the following line to the path to image0 label0
path_to_image0_label0 = r"logs/learn2reg_t1_paired_train_logs/test"
path_to_pred_fixed_img = os.path.join(path_to_image0_label0, r"pair_0/moving_image")
path_to_moving_img = os.path.join(path_to_image0_label0, r"pair_0/pred_fixed_image")

# change inds_to_plot if different images need to be plotted instead

inds_to_plot = [144, 145, 184, 34, 90, 21]
sub_plot_counter = 1

for ind in inds_to_plot:
    plt.subplot(6, 2, sub_plot_counter)
    label = plt.imread(
        os.path.join(path_to_moving_img, "depth" + str(ind) + "_moving_image.png")
    )
    plt.imshow(label)
    plt.axis("off")
    if sub_plot_counter == 1:
        plt.title("Moving Image")

    plt.subplot(6, 2, sub_plot_counter + 1)
    pred = plt.imread(
        os.path.join(
            path_to_pred_fixed_img, "depth" + str(ind) + "_pred_fixed_image.png"
        )
    )
    plt.imshow(pred)
    plt.axis("off")
    if sub_plot_counter == 1:
        plt.title("Warped Fixed Image")

    sub_plot_counter = sub_plot_counter + 2

path_to_vis = r"logs/learn2reg_t1_paired_train_logs/visualisation.png"
plt.savefig(path_to_vis)
print("Visualisation saved to:", path_to_vis)
