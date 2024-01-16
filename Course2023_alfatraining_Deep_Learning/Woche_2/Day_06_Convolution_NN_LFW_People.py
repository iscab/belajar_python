# Python, using Anaconda environment
# Week 1, Day 6

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# Die Bilder herunterladen
data = fetch_lfw_people(data_home=None,
                        funneled=True,
                        resize=1,  # Im Falle von Arbeitsspeicherproblemen: 0.5
                        min_faces_per_person=50,
                        color=True,
                        slice_=(slice(50, 200, None), slice(50, 200, None)),
                        # slice_=(slice(0, 250, None), slice(0, 250, None)),

                        download_if_missing=True,
                        return_X_y=False)

# Data structure
print(type(data), data.keys())

# In x, y aufteilen
x = data.images
y = data.target


# Bilder normalisieren
# Je nach scikit-learn-Version ist x.max() entweder 1 oder 255
# Wir geben uns erst mal die Max- und Min-Werte der Pixelintensitäten aus
print('Max and Min pixel values:', x.max(), x.min(), "\n")
print("Input size and type:  ", x.shape, type(x))
image_size = (x.shape[1:4])
print("RGB image size:  ", image_size, type(image_size))
# exit()

# Die Pixel sollten am besten alle mit dem gleichen Faktor skaliert werden
# Mit dem MinMaxscaler würden wir pro Pixel einen unterschielichen Skalierungsfaktor
# bekommen und dadurch werden die Bilder sehr scheckig
# manche Ecken oder Kanten würden in andere Bilder einfließen
x = x/x.max()


# Einige Bilder darstellen
plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)  # Subplot mit 3 Zeilen und 4 Spalten
    plt.imshow(x[y == i][0])
    plt.title(data.target_names[i], fontsize=8)
    plt.axis('off')  # Keine Achsenbeschriftung
# plt.show()

# Labels
print('Max and Min label values:', max(y), min(y), "\n")
print("Labels:  \n", data.target_names.shape, data.target_names, type(data.target_names))
num_label = data.target_names.shape[0]
print(num_label, type(num_label))

# Labels as unit vectors (OneHotEncode)
enc = OneHotEncoder()
enc.fit(y.reshape(-1, 1))
y_enc = enc.transform(y.reshape(-1, 1)).toarray()

# print(y_enc.shape, type(y_enc))
# print(y[:5], y_enc[:5,:])

# train and test split
X_train, X_test, y_train, y_test = train_test_split(x, y_enc, test_size=0.2, random_state=42)

print("Train data size and type:  \n", X_train.shape, type(X_train), y_train.shape, type(y_train))
print("Test data size and type:  \n", X_test.shape, type(X_test), y_test.shape, type(y_test))


# Convolutional Neural Network (CNN) model
cnn_model = tf.keras.models.Sequential()
regularizer = tf.keras.regularizers.L2(0.01)
cnn_model.add(tf.keras.layers.Conv2D(32, (5, 5), input_shape=image_size, activation="relu"))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(tf.keras.layers.Flatten())
cnn_model.add(tf.keras.layers.Dense(64, activation="tanh", kernel_regularizer=regularizer))
cnn_model.add(tf.keras.layers.Dense(32, activation="sigmoid", kernel_regularizer=regularizer))
cnn_model.add(tf.keras.layers.Dense(num_label, activation="softmax"))

cnn_model.summary()

# Optimizer mit bestimmter Lernrate festlegen
opt = tf.keras.optimizers.Adam(learning_rate=0.001)  # 0.01

# Loss-Funktion, Optimizer und Metrik dem neuronalen Netz zuweisen
cnn_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

# das Modell trainieren
history = cnn_model.fit(X_train, y_train, batch_size=50,
                    epochs=100,
                    validation_data=(X_test, y_test))

exit()

# show Figures
plt.show()

# end of file
