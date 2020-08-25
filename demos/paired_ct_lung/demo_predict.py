import os

import matplotlib.pyplot as plt

from deepreg.predict import predict

######## PREDICTION ########


log_dir = "learn2reg_t2_paired_train_logs"

log_dir_tr = r"demos/paired_ct_lung/learn2reg_t2_paired_train_logs"
ckpt_path = os.path.join(log_dir_tr, "save", "weights-epoch500.ckpt")
config_path = os.path.join(log_dir_tr, "config.yaml")

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
    save_png=True,
)

# the numerical metrics are saved in the logs directory specified


######## VISUALISATION ########

# Now lets load in a few samples from the predicitons and plot them

# change the following line to the path to image0 label0

path_to_test = r"logs/learn2reg_t2_paired_train_logs/test"
path_to_fixed_label = os.path.join(path_to_test, "pair_0", "label_0", "fixed_label")
path_to_pred_fixed_label = os.path.join(
    path_to_test, "pair_0", "label_0", "pred_fixed_label"
)
path_to_fixed_image = os.path.join(path_to_test, "pair_0", "fixed_image")
path_to_pred_fixed_image = os.path.join(path_to_test, "pair_0", "pred_fixed_image")
path_to_moving_image = os.path.join(path_to_test, "pair_0", "moving_image")
path_to_moving_label = os.path.join(path_to_test, "pair_0", "label_0", "moving_label")


# change inds_to_plot if different images need to be plotted instead

inds_to_plot = [20, 80, 110, 170, 190, 200]
sub_plot_counter = 1

for ind in inds_to_plot:
    plt.subplot(6, 6, sub_plot_counter)
    label = plt.imread(
        os.path.join(path_to_fixed_label, "depth" + str(ind) + "_fixed_label.png")
    )
    plt.imshow(label)
    plt.axis("off")
    if sub_plot_counter == 1:
        plt.title("fixed_label")

    plt.subplot(6, 6, sub_plot_counter + 1)
    pred = plt.imread(
        os.path.join(
            path_to_pred_fixed_label, "depth" + str(ind) + "_pred_fixed_label.png"
        )
    )
    plt.imshow(pred)
    plt.axis("off")
    if sub_plot_counter == 1:
        plt.title("pred_fixed_label")

    plt.subplot(6, 6, sub_plot_counter + 2)
    fixed_im = plt.imread(
        os.path.join(path_to_fixed_image, "depth" + str(ind) + "_fixed_image.png")
    )
    plt.imshow(fixed_im)
    plt.axis("off")
    if sub_plot_counter == 1:
        plt.title("fixed_image")

    plt.subplot(6, 6, sub_plot_counter + 3)
    pr_fixed_im = plt.imread(
        os.path.join(
            path_to_pred_fixed_image, "depth" + str(ind) + "_pred_fixed_image.png"
        )
    )
    plt.imshow(pr_fixed_im)
    plt.axis("off")
    if sub_plot_counter == 1:
        plt.title("pred_fixed_image")

    plt.subplot(6, 6, sub_plot_counter + 4)
    mov_im = plt.imread(
        os.path.join(path_to_moving_image, "depth" + str(ind) + "_moving_image.png")
    )
    plt.imshow(mov_im)
    plt.axis("off")
    if sub_plot_counter == 1:
        plt.title("moving_image")

    plt.subplot(6, 6, sub_plot_counter + 5)
    mov_l = plt.imread(
        os.path.join(path_to_moving_label, "depth" + str(ind) + "_moving_label.png")
    )
    plt.imshow(mov_l)
    plt.axis("off")
    if sub_plot_counter == 1:
        plt.title("moving_label")

    sub_plot_counter = sub_plot_counter + 6

path_to_vis = r"logs/learn2reg_t2_paired_train_logs/visualisation.png"
plt.savefig(path_to_vis)
print("Visualisation saved to:", path_to_vis)
