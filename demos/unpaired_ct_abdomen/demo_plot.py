import os
import shutil
import matplotlib.pyplot as plt

# Plot results from predictions

log_dir = "unpaired_ct_abdomen_log"
images_dir = os.path.join("logs", log_dir, "test/pair_0_1")

# Delete old plots

plot_folder = os.path.join("logs", log_dir, "plot")
if os.path.exists(plot_folder):
    shutil.rmtree(plot_folder) # delete old data

# Plotting results of pair_0_1 and label 1 (testing)
plt.figure(1)
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
os.mkdir(plot_folder)
plot_dir = os.path.join(plot_folder, "results_pair_0_1_label_1.png")
print(plot_dir)
plt.savefig(plot_dir)

# Plotting results of pair_0_1 and label 5 (testing)
plt.figure(2)
plt.subplot(2, 3, 1)
label_5_fixed_image = plt.imread(os.path.join(images_dir,"fixed_image","depth69_fixed_image.png"))
plt.imshow(label_5_fixed_image)
plt.title("Label 5 fixed image")
plt.axis("off")

plt.subplot(2, 3, 2)
label_5_fixed_label = plt.imread(os.path.join(images_dir,"label_5/fixed_label","depth69_fixed_label.png"))
plt.imshow(label_5_fixed_label)
plt.title("Label 5 fixed label")
plt.axis("off")

plt.subplot(2, 3, 3)
label_5_predicted = plt.imread(os.path.join(images_dir,"label_5/pred_fixed_label","depth69_pred_fixed_label.png"))
plt.imshow(label_5_predicted)
plt.title("Label 5 prediction")
plt.axis("off")

plt.subplot(2, 3, 4)
label_5_moving_image = plt.imread(os.path.join(images_dir,"moving_image","depth69_moving_image.png"))
plt.imshow(label_5_moving_image)
plt.title("Label 5 moving image")
plt.axis("off")

plt.subplot(2, 3, 5)
label_5_moving_label = plt.imread(os.path.join(images_dir,"label_5/moving_label","depth56_moving_label.png"))
plt.imshow(label_5_moving_label)
plt.title("Label 5 moving label")
plt.axis("off")

# Save image
plot_dir = os.path.join(plot_folder, "results_pair_0_1_label_5.png")
print(plot_dir)
plt.savefig(plot_dir)

print("All results saved in:", plot_dir)