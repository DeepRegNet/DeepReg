import os

import matplotlib.pyplot as plt

# Plot results from predictions

log_dir = "unpaired_ct_abdomen_log"
images_dir = os.path.join("logs", log_dir, "test/pair_0_1")

# Plotting results of cases 1 and 2 (testing)

plt.subplot(2, 3, 1)
label_1_fixed_image = plt.imread(os.path.join(images_dir,"fixed_image","depth56_fixed_image.png"))
plt.imshow(label_1_fixed_image)
plt.title("Label 1 fixed image")
plt.axis("off")

plt.subplot(2, 3, 2)
label_1_fixed_label = plt.imread(os.path.join(images_dir,"label_1/fixed_label","depth56_fixed_label.png"))
plt.imshow(label_1_fixed_label)
plt.title("Label 1 fixed label")
plt.axis("off")

plt.subplot(2, 3, 3)
label_1_predicted = plt.imread(os.path.join(images_dir,"label_1/pred_fixed_label","depth56_pred_fixed_label.png"))
plt.imshow(label_1_predicted)
plt.title("Label 1 prediction")
plt.axis("off")

plt.subplot(2, 3, 4)
label_1_moving_image = plt.imread(os.path.join(images_dir,"moving_image","depth56_moving_image.png"))
plt.imshow(label_1_moving_image)
plt.title("Label 1 moving image")
plt.axis("off")

plt.subplot(2, 3, 5)
label_1_moving_label = plt.imread(os.path.join(images_dir,"label_1/moving_label","depth56_moving_label.png"))
plt.imshow(label_1_moving_label)
plt.title("Label 1 moving label")
plt.axis("off")

# Save image
os.mkdir(os.path.join("logs", log_dir, "plot"))
plot_dir = os.path.join("logs", log_dir, "plot", "results_label1_label2.png")
print(plot_dir)
plt.savefig(plot_dir)
print("Results saved in:", plot_dir)