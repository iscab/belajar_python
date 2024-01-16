""" Image Reconstruction, especially Super-Resolution

Procedure of data processing and CNN-model fitting:
1. load data
2. prepare training and validation/test data
3. prepare Convolutional Neural Network (CNN) model
4. model fitting
5. model evaluation

data sources:  https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution/data

environment: Python 3.11 and TensorFlow 2.15.0 under Anaconda 23.7.4

file version: 19:59 04.01.2024

"""

__author__ = "Ignatius S. Condro Atmawan"
__contact__ = "saptocondro@gmail.com"
__copyright__ = "alfatraining"
__date__ = "2024/01/05"

import os
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

# random seed
random_seed = 42

# prepare output folder
current_directory = os.getcwd()
output_directory = os.path.join(current_directory, "results")
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)
# time stamp format for output files
standard_date_time_format = "_%Y_%m_%d_%H_%M_%S"


# loading data
batch_size = 32
image_size = 256  # original image has 256 x 256 pixels
n_downscale = 1  # 1 or 2
low_size = int(image_size/pow(2, n_downscale))
percent_resize = 100.0 * low_size / image_size


train_data_high = tf.keras.utils.image_dataset_from_directory(
    r"dataset\train\high_res",
    labels=None,
    label_mode=None,
    batch_size=batch_size,
    color_mode="rgb",
    image_size=(image_size, image_size),
    shuffle=False,
    follow_links=False,
    crop_to_aspect_ratio=False
)
train_data_low = tf.keras.utils.image_dataset_from_directory(
    r"dataset\train\high_res",
    labels=None,
    label_mode=None,
    batch_size=batch_size,
    color_mode="rgb",
    image_size=(low_size, low_size),
    interpolation="bilinear",
    shuffle=False,
    follow_links=False,
    crop_to_aspect_ratio=False
)
# print(train_data_high, type(train_data_high))
# print(train_data_low, type(train_data_low))
# exit()

test_data_high = tf.keras.utils.image_dataset_from_directory(
    r"dataset\val\high_res",
    labels=None,
    label_mode=None,
    batch_size=batch_size,
    color_mode="rgb",
    image_size=(image_size, image_size),
    shuffle=False,
    follow_links=False,
    crop_to_aspect_ratio=False
)
test_data_low = tf.keras.utils.image_dataset_from_directory(
    r"dataset\val\high_res",
    labels=None,
    label_mode=None,
    batch_size=batch_size,
    color_mode="rgb",
    image_size=(low_size, low_size),
    interpolation="bilinear",
    shuffle=False,
    follow_links=False,
    crop_to_aspect_ratio=False
)
print("\n")


def normalize(x):
    """
    Normalize the images from int to float between -1 and 1

    :param x: Image
    :return: normalized image
    """
    x = tf.keras.layers.Rescaling(scale=1/255, offset=0)(x)
    return x


def PSNR(high_resolution_ground_truth, super_resolution):
    """
    Peak signal-to-noise ratio (PSNR)
    link: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    :param high_resolution_ground_truth: desired Image
    :param super_resolution: generated Image
    :return: (float) the PSNR value
    """
    high_resolution_ground_truth = high_resolution_ground_truth * 255
    super_resolution = super_resolution * 255
    psnr_value = tf.image.psnr(high_resolution_ground_truth, super_resolution, max_val=255)[0]
    return psnr_value
# mse, mae

def SSIM(high_resolution_ground_truth, super_resolution):
    """
    Structural similarity index measure (SSIM)
    link: https://en.wikipedia.org/wiki/Structural_similarity

    :param high_resolution_ground_truth: desired Image
    :param super_resolution: generated Image
    :return: (float) the SSIM value, between -1 and 1
    """
    high_resolution_ground_truth = high_resolution_ground_truth * 255
    super_resolution = super_resolution * 255
    ssim_value = tf.image.ssim(high_resolution_ground_truth, super_resolution, max_val=255)[0]
    return ssim_value


# Image visualization
high_batch = train_data_high.take(1)
low_batch = train_data_low.take(1)
select_image = 5  # smaller than batch size
plt.figure()
for images in high_batch:
    ax = plt.subplot(1, 2, 1)
    plt.imshow(images[select_image].numpy().astype('int'))
    ax.set_title("ground truth")
    # plt.axis("off")
for images in low_batch:
    ax = plt.subplot(1, 2, 2)
    plt.imshow(images[select_image].numpy().astype('int'))
    ax.set_title(f"lower resolution: {percent_resize} %")
    # plt.axis("off")
plt.suptitle("Pair images with different resolutions")
plt.tight_layout()
# saving plots
ticktime = dt.datetime.now()
file_name_stamp = ticktime.strftime(standard_date_time_format)
file_name_png = "training_ground_truth_vs_low_resolution"
file_name_png += f"_{int(percent_resize)}_percent" + file_name_stamp + ".png"
file_name_png = os.path.join(output_directory, file_name_png)
# print(file_name_png)
plt.savefig(file_name_png)
# plt.show()
# exit()

# Residual Block
n_kernel = 32  # 32 or 64
residual_cell = tf.keras.models.Sequential()
residual_cell.add(tf.keras.layers.Conv2D(n_kernel, (3,3), padding="same", activation="relu"))
residual_cell.add(tf.keras.layers.BatchNormalization())
residual_cell.add(tf.keras.layers.Conv2D(n_kernel, (3,3), padding="same"))

# skip connection
input_ = tf.keras.layers.Input(shape=(image_size, image_size, n_kernel))
hidden1 = residual_cell(input_)
hidden_add = tf.keras.layers.Add()([input_, hidden1])
out = tf.keras.layers.ReLU()(hidden_add)

residual_blocks = []
n_block = 4  # try 1, 2, 4, 8
for idx in range(n_block):
    residual_block = tf.keras.models.Model(inputs=[input_], outputs=[out])
    residual_block._name = f"residual_network_{idx + 1}"
    residual_blocks.append(residual_block)
residual_blocks[0].summary()
# residual_blocks[1].summary()
# exit()


# Image Reconstruction Model
want_residual_block = True  # do you want residual block? True or false
want_residual_block = want_residual_block and n_block >= 1
leaky_faktor = 0.2
reconstruction_model = tf.keras.models.Sequential()
reconstruction_model.add(tf.keras.layers.Conv2D(n_kernel, (5,5), padding="same", activation="relu",
                                                input_shape=(low_size, low_size, 3)))

# upscaling
for idx in range(n_downscale):
    reconstruction_model.add(tf.keras.layers.Conv2DTranspose(n_kernel, (3,3), strides=(2,2), padding="same"))
    reconstruction_model.add(tf.keras.layers.BatchNormalization())
    reconstruction_model.add(tf.keras.layers.LeakyReLU(leaky_faktor))

# residual network
if want_residual_block:
    for idx in range(n_block):
        reconstruction_model.add(residual_blocks[idx])

reconstruction_model.add(tf.keras.layers.Conv2D(64, (5,5), padding="same", activation="relu"))
reconstruction_model.add(tf.keras.layers.BatchNormalization())
reconstruction_model.add(tf.keras.layers.LeakyReLU(leaky_faktor))

reconstruction_model.add(tf.keras.layers.Conv2D(3, (3,3), padding="same", activation="relu"))
reconstruction_model._name = "reconstruction_model"

# optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-4, 5e-5]))

# compile Model
reconstruction_model.compile(loss="mae", optimizer=optimizer, metrics=[PSNR, SSIM, "mse", "mae"])

reconstruction_model.summary()
# exit()

# normalize image (for plotting, multiply them with 255)
train_data_high = train_data_high.map(normalize)
train_data_low = train_data_low.map(normalize)
test_data_high = test_data_high.map(normalize)
test_data_low = test_data_low.map(normalize)

# train_data_low = train_data_low.unbatch()
# train_data_high = train_data_high.unbatch()

# history = reconstruction_model.fit(train_data_low, train_data_high, batch_size=batch_size, epochs=30)
history_mse = []
history_mae = []
history_PSNR = []
history_SSIM = []
n_epoch = 20
for idx in range(1, n_epoch+1):
    print("-"*15, "Epoch %d" % idx, "-"*15, "\n")
    history = reconstruction_model.fit(zip(train_data_low, train_data_high), batch_size=batch_size, verbose=True)
    history_mse.append(history.history["mse"])
    history_mae.append(history.history["mae"])
    history_PSNR.append(history.history["PSNR"])
    history_SSIM.append(history.history["SSIM"])


# Model evaluation
print("Model evaluation:  ")
test_score = reconstruction_model.evaluate(zip(test_data_low, test_data_high), verbose=0)
# print(test_score, type(test_score))
test_PSNR = test_score[1]
test_SSIM = test_score[2]
test_MSE = test_score[3]
test_MAE = test_score[4]
# exit()

# Visualization of metrics, during learning/fitting progress and info about test score
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(history_mse, label=f"mse,  test: {test_MSE:.2f}")
plt.plot(history_mae, label=f"mae,  test: {test_MAE:.2f}")
plt.xlabel("epoch")
plt.legend(loc="upper left")
plt.subplot(3, 1, 2)
plt.plot(history_PSNR, label=f"PSNR,  test: {test_PSNR:.2f}")
plt.xlabel("epoch")
plt.legend(loc="upper left")
plt.subplot(3, 1, 3)
plt.plot(history_SSIM, label=f"SSIM,  test: {test_SSIM:.2f}")
plt.xlabel("epoch")
plt.legend(loc="upper left")
plt.suptitle(f"Image Restoration progress, {n_epoch} epochs")
plt.tight_layout()
# saving plots
ticktime = dt.datetime.now()
file_name_stamp = ticktime.strftime(standard_date_time_format)
file_name_png = f"restoration_metrics_vs_{n_epoch}_epoch"
file_name_png += file_name_stamp + ".png"
file_name_png = os.path.join(output_directory, file_name_png)
# print(file_name_png)
plt.savefig(file_name_png)
# plt.show()

generated_images = reconstruction_model.predict(test_data_low)
generated_images = ((generated_images)*255).astype("int")

high_batch = test_data_high.take(1)
low_batch = test_data_low.take(1)
select_image = 8
plt.figure()
for images in high_batch:
    ax = plt.subplot(1, 3, 3)
    image = images[select_image].numpy() * 255
    plt.imshow(image.astype('int'))
    ax.set_title("ground truth")
    # plt.axis("off")
for images in low_batch:
    ax = plt.subplot(1, 3, 1)
    image = images[select_image].numpy() * 255
    plt.imshow(image.astype('int'))
    ax.set_title(f"lower resolution: {percent_resize} %")
    # plt.axis("off")
ax = plt.subplot(1, 3, 2)
plt.imshow(generated_images[select_image])
ax.set_title("generated image")
# plt.axis("off")
plt.suptitle(f"Image Restoration Results, {n_epoch} epochs")
plt.tight_layout()
# saving plots
ticktime = dt.datetime.now()
file_name_stamp = ticktime.strftime(standard_date_time_format)
file_name_png = f"restoration_result_{n_epoch}_epoch_from_low_resolution"
file_name_png += f"_{int(percent_resize)}_percent" + file_name_stamp + ".png"
file_name_png = os.path.join(output_directory, file_name_png)
# print(file_name_png)
plt.savefig(file_name_png)
# plt.show()

# print(generated_images.shape, type(generated_images))

plt.show()

# end of file, version: 19:59 04.01.2024
