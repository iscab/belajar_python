""" Image Reconstruction, especially Super-Resolution

Procedure of data processing and CNN-model fitting:
1. load data
2. prepare training and validation/test data
3. prepare Convolutional Neural Network (CNN) model
4. model fitting
5. model evaluation

data sources:  https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution/data

environment: Python 3.11 and TensorFlow 2.15.0 under Anaconda 23.7.4

file version: 08:48 03.01.2024

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
# dataset\train\high_res
# dataset\train\low_res
train_data = tf.keras.utils.image_dataset_from_directory(
    r"dataset\train",
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    color_mode="rgb",
    image_size=(256, 256),
    shuffle=False,
    seed=random_seed,
    follow_links=False,
    crop_to_aspect_ratio=False
)

print(train_data, type(train_data))

"""dummy_data = train_data.unbatch()
dummy_data = dummy_data.batch(batch_size, drop_remainder=True)

for images in dummy_data.take(1):
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
plt.show()"""

"""a=0
for x, y in train_data.take(1):
    print(x.shape, type(x))
    print(y, type(y))
    a = a + 1
print(a)"""

my_shape = np.array([])
for x, y in train_data:
    print(x.shape)
print(my_shape)






# end of file, version: 08:48 03.01.2024
