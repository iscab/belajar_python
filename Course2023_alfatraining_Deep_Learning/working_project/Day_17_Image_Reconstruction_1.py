""" Image Reconstruction, especially Super-Resolution

Procedure of data processing and CNN-model fitting:
1. load data
2. prepare training and validation/test data
3. prepare Convolutional Neural Network (CNN) model
4. model fitting
5. model evaluation

data sources:  https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution/data

environment: Python 3.11 and TensorFlow 2.15.0 under Anaconda 23.7.4

file version: 10:50 03.01.2024

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
# dataset\train\high_res
# dataset\train\low_res
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

print(train_data_high, type(train_data_high))
print(train_data_low, type(train_data_low))

plt.figure()
for images in train_data_high.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")

plt.figure()
for images in train_data_low.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")

plt.show()



# end of file, version: 10:50 03.01.2024
