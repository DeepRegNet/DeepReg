import os

import matplotlib.pyplot as plt

from deepreg.predict import predict

######## PREDICTION ########

log_dir = "/home/acasamitjana/Repositories/DeepRegNet/DeepReg/demos/mr-brain-t4/learn2reg_t4_unpaired_train_logs"
config_path = ["/home/acasamitjana/Repositories/DeepRegNet/DeepReg/demos/mr-brain-t4/mr_brain_t4.yaml"]
ckpt_path = os.path.join("logs", log_dir, "save", "weights-epoch200.ckpt")
gpu = 'none'
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

exit()
# the numerical metrics are saved in the logs directory specified



######## VISUALISATION ########

# Now lets load in a few samples from the predicitons and plot them

# change the following line to the path to image0 label0
path_to_image0_label0 = r""
os.chdir(path_to_image0_label0)

# change image names if different images need to be plotted instead

plt.subplot(3, 2, 1)
label144 = plt.imread("depth144_fixed_label.png")
plt.imshow(label144)
plt.title("Label")
plt.axis("off")

plt.subplot(3, 2, 2)
pred144 = plt.imread("depth144_fixed_pred.png")
plt.imshow(pred144)
plt.title("Prediction")
plt.axis("off")


plt.subplot(3, 2, 3)
label145 = plt.imread("depth145_fixed_label.png")
plt.imshow(label145)
plt.axis("off")

plt.subplot(3, 2, 4)
pred145 = plt.imread("depth145_fixed_pred.png")
plt.imshow(pred145)
plt.axis("off")


plt.subplot(3, 2, 5)
label184 = plt.imread("depth184_fixed_label.png")
plt.imshow(label184)
plt.axis("off")

plt.subplot(3, 2, 6)
pred184 = plt.imread("depth184_fixed_pred.png")
plt.imshow(pred184)
plt.axis("off")

# this is the path where you want to save the visualisation as a png
path_to_save_fig = r""
plt.savefig(os.path.join(path_to_save_fig, "labels_and_preds.png"))

print("Visual representation of predictions saved to path specified")
