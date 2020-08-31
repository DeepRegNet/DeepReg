import os

import matplotlib.pyplot as plt

from deepreg.predict import predict

######## PREDICTION ########
# saved weights: change it if you retrained the model.
log_dir = r"learn2reg_t4_unpaired_train_logs"
ckpt_path = os.path.join("logs", log_dir, "save", "weights-epoch20.ckpt")
config_path = [os.path.join("logs", log_dir, "config.yaml")]
gpu = "0"
gpu_allow_growth = False
predict(
    gpu=gpu,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    mode="test",
    batch_size=1,
    log_dir=log_dir,
    sample_label="all",
    config_path=config_path,
)

# the numerical metrics are saved in the logs directory specified


######## VISUALISATION ########

# Now lets load in a few samples from the predictions and plot them
main_path = os.getcwd()
os.chdir(main_path)

# change the following line to the path to image0 label0
path_to_image0_label0 = os.path.join("logs", log_dir, "test")
path_to_image0_label0 = os.path.join(main_path, path_to_image0_label0)

path_to_pred_fixed_label = os.path.join(
    path_to_image0_label0, r"pair_0_1/label_0/pred_fixed_label"
)
path_to_fixed_label = os.path.join(
    path_to_image0_label0, r"pair_0_1/label_0/fixed_label"
)

path_to_fixed_image = os.path.join(path_to_image0_label0, "pair_0_1", "fixed_image")
path_to_pred_fixed_image = os.path.join(
    path_to_image0_label0, "pair_0_1", "pred_fixed_image"
)
path_to_moving_image = os.path.join(path_to_image0_label0, "pair_0_1", "moving_image")
path_to_moving_label = os.path.join(
    path_to_image0_label0, "pair_0_1", "label_0", "moving_label"
)

# change inds_to_plot if different images need to be plotted instead

inds_to_plot = [10, 15, 20, 25, 30, 35]
sub_plot_counter = 1
plt.rcParams["figure.figsize"] = [20, 20]

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

path_to_vis = os.path.join(main_path, "logs", log_dir, "visualisation.png")
plt.savefig(path_to_vis)
print("Visualisation saved to:", path_to_vis)
