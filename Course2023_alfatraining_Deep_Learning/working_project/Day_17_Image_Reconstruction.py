""" Image Reconstruction, especially Super-Resolution

Procedure of data processing and CNN-model fitting:
1. load data
2. prepare training and validation/test data
3. prepare Convolutional Neural Network (CNN) model
4. model fitting
5. model evaluation

data sources:  https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution/data

environment: Python 3.11 and TensorFlow 2.15.0 under Anaconda 23.7.4

file version: 12:02 03.01.2024

"""

__author__ = "Ignatius S. Condro Atmawan"
__contact__ = "saptocondro@gmail.com"
__copyright__ = "alfatraining"
__date__ = "2024/01/05"

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

# random seed
random_seed = 42

# loading data
batch_size = 32
image_size = (256, 256)  # original image has 256 x 256 pixels

train_data_high = tf.keras.utils.image_dataset_from_directory(
    r"dataset\train\high_res",
    labels=None,
    label_mode=None,
    batch_size=batch_size,
    color_mode="rgb",
    image_size=image_size,
    shuffle=False,
    follow_links=False,
    crop_to_aspect_ratio=False
)
train_data_low = tf.keras.utils.image_dataset_from_directory(
    r"dataset\train\low_res",
    labels=None,
    label_mode=None,
    batch_size=batch_size,
    color_mode="rgb",
    image_size=image_size,
    shuffle=False,
    follow_links=False,
    crop_to_aspect_ratio=False
)
# print(train_data_high, type(train_data_high))
# print(train_data_low, type(train_data_low))

test_data_high = tf.keras.utils.image_dataset_from_directory(
    r"dataset\val\high_res",
    labels=None,
    label_mode=None,
    batch_size=batch_size,
    color_mode="rgb",
    image_size=image_size,
    shuffle=False,
    follow_links=False,
    crop_to_aspect_ratio=False
)
test_data_low = tf.keras.utils.image_dataset_from_directory(
    r"dataset\val\low_res",
    labels=None,
    label_mode=None,
    batch_size=batch_size,
    color_mode="rgb",
    image_size=image_size,
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
    """"""
    psnr_value = tf.image.psnr(high_resolution_ground_truth, super_resolution, max_val=255)[0]
    return psnr_value
# mse, mae

# Image visualization
high_batch = train_data_high.take(1)
low_batch = train_data_low.take(1)
select_image = 5  # smaller than batch size
plt.figure()
for images in high_batch:
    ax = plt.subplot(1, 2, 1)
    plt.imshow(images[select_image].numpy().astype('int'))
    ax.set_title("ground truth")
    plt.axis("off")
for images in low_batch:
    ax = plt.subplot(1, 2, 2)
    plt.imshow(images[select_image].numpy().astype('int'))
    ax.set_title("lower resolution")
    plt.axis("off")
plt.suptitle("Pair images with different resolutions")
# plt.show()
# exit()

# Residual Block
# residual_cell = tf.keras.models.Sequential()



# Image Reconstruction Model
reconstruction_model = tf.keras.models.Sequential()
reconstruction_model.add(tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu",
                                                input_shape=(256, 256, 3)))
# reconstruction_model.add(Residual(32, (3,3)))
# reconstruction_model.add(Residual(32, (3,3)))
reconstruction_model.add(tf.keras.layers.Conv2D(64, (5,5), padding="same", activation="relu"))
reconstruction_model.add(tf.keras.layers.Conv2D(3, (3,3), padding="same", activation="relu"))

# optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-4, 5e-5]))

# compile Model
reconstruction_model.compile(loss="mae", optimizer=optimizer, metrics=[PSNR])

reconstruction_model.summary()

# normalize image (for plotting, multiply them with 255)
train_data_high = train_data_high.map(normalize)
train_data_low = train_data_low.map(normalize)
test_data_high = test_data_high.map(normalize)
test_data_low = test_data_low.map(normalize)

# train_data_low = train_data_low.unbatch()
# train_data_high = train_data_high.unbatch()

# history = reconstruction_model.fit(train_data_low, train_data_high, batch_size=batch_size, epochs=30)
n_epoch = 10
history_rec = []
for idx in range(1, n_epoch+1):
    print('-'*15, 'Epoch %d' % idx, '-'*15)
    history = reconstruction_model.fit(zip(train_data_low, train_data_high), batch_size=batch_size, verbose=True)
    history_rec.append(history.history["loss"])

plt.figure()
plt.plot(history_rec, label='Reconstruction loss')
# plt.show()

# Model evaluation
print("Model evaluation:  ")
generated_images = reconstruction_model.predict(test_data_low)
generated_images = ((generated_images)*255).astype(int)

high_batch = test_data_high.take(1)
low_batch = test_data_low.take(1)
select_image = 8
plt.figure()
for images in high_batch:
    ax = plt.subplot(1, 3, 3)
    image = images[select_image].numpy() * 255
    plt.imshow(image.astype('int'))
    ax.set_title("ground truth")
    plt.axis("off")
for images in low_batch:
    ax = plt.subplot(1, 3, 1)
    image = images[select_image].numpy() * 255
    plt.imshow(image.astype('int'))
    ax.set_title("lower resolution")
    plt.axis("off")
ax = plt.subplot(1, 3, 2)
plt.imshow(generated_images[select_image])
ax.set_title("generated image")
plt.axis("off")
plt.suptitle("Results")

print(generated_images.shape, type(generated_images))

#subclassing



plt.show()

# end of file, version: 12:02 03.01.2024
