# Python, using Anaconda environment
# Week 1, Day 8

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

def r2_score(y_true, y_pred):
    y_diff = tf.reduce_sum(tf.square(y_true-y_pred))
    y_square = tf.reduce_sum(tf.square(y_true-tf.reduce_mean(y_true)))
    return 1-y_diff/y_square

train_data = keras.utils.image_dataset_from_directory(
    r'C:\Users\alfa\PycharmProjects\python_deeplearning\Woche_2\Bilder Wuerfel 2 small train',  # Pfad zu den Bildern
    labels='inferred',            # Labels aus den Dateiname ableiten
    # 'int' oder 'categorical' je nach Aufgabe Regression oder Klassifikation
    label_mode='int',
    class_names=None,             # Man kann optional eine Liste der Klassennamen übergeben
    color_mode='rgb',             # Bilder farbig ausgeben. Alternativ: 'grayscale', 'rgba'
    # Wie viele Bilder auf einmal von der Festplatte geladen werden
    batch_size=32,
    image_size=(224, 224),        # Biler auf die angegebene Größe sklaieren
    shuffle=True,                 # Vor jeder Epoche die Bilder durchmischen
    # if not None Durchmischen in bestimmter Reihenfolge durchführen
    seed=0,
    validation_split=0.2,         # Gibt den Anteil der Testdaten an
    # Gibt an, ob man die Trainigs- oder Testdaten bekommen möchte
    subset='training',
    interpolation='bilinear',     # Wie die Skalierung durchfgeführt wird
    follow_links=False,           # Ordnerstruktur nachverfolgen
    crop_to_aspect_ratio=False)   # Auf die Bildmitte zuschneiden um das Seitenverhältnis nicht zu verändern

test_data = keras.utils.image_dataset_from_directory(
    r'C:\Users\alfa\PycharmProjects\python_deeplearning\Woche_2\Bilder Wuerfel 2 small test',  # Pfad zu den Bildern
    labels='inferred',            # Labels aus den Dateiname ableiten
    # 'int' oder 'categorical' je nach Aufgabe Regression oder Klassifikation
    label_mode='int',
    class_names=None,             # Man kann optional eine Liste der Klassennamen übergeben
    color_mode='rgb',             # Bilder farbig ausgeben. Alternativ: 'grayscale', 'rgba'
    # Wie viele Bilder auf einmal von der Festplatte geladen werden
    batch_size=32,
    image_size=(224, 224),        # Biler auf die angegebene Größe sklaieren
    shuffle=True,                 # Vor jeder Epoche die Bilder durchmischen
    # if not None Durchmischen in bestimmter Reihenfolge durchführen
    seed=None,
    validation_split=None,        # Gibt den Anteil der Testdaten an
    # Gibt an, ob man die Trainigs- oder Testdaten bekommen möchte
    subset=None,
    interpolation='bilinear',     # Wie die Skalierung durchfgeführt wird
    follow_links=False,           # Ordnerstruktur nachverfolgen
    crop_to_aspect_ratio=False)   # Auf die Bildmitte zuschneiden um das Seitenverhältnis nicht zu verändern

# Einige Bilder der Trainingsdaten abrufen und anzeigen
# Mit dem take Befehl können wir aus den Trainigsdaten einen nächsten Batch abrufen
batch = train_data.take(1)
print("\n")

# Die Dimensionen der Bilder ausgeben, dazu den Batch in eine Liste umwandeln
# Wir geben für den ersten Batch (erste [0]), und die Bilder [zweite [0]] den shape aus
print(list(batch)[0][0].shape)

# Dimensionen der Labels des Batches
print(list(batch)[0][1].shape)


# Um an die Daten ranzukommen kann auch über den gesamten Datensatz iteriert werden
plt.figure(figsize=(24, 18))
for images, labels in train_data:
    print(images.shape, type(images))
    print(labels.shape, type(labels))
    for i in range(32):
        plt.subplot(4, 8, i+1)
        plt.imshow(images[i].numpy().astype('int'))
        plt.title(train_data.class_names[labels[i]], fontsize=18)
        plt.axis('off')
    # plt.show()
    break

# prepare inputs for MobileNet
"""inputs = keras.applications.mobilenet.preprocess_input(train_data)
print(train_data.cardinality().numpy())
exit()"""

# Model: MobileNet
"""cnn_model = keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.001,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling="avg",
    classes=30,
    classifier_activation=None,
)  # functional"""
# cnn_model.add(keras.layers.Dense(30, activation="relu")) then add output

# TODO: sequential model, and then add (keras.applications.MobileNet) without classes
cnn_model = keras.models.Sequential()
regularizer = keras.regularizers.L2(0.01)
cnn_model.add(keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.001,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling="avg",
    # classes=30,
    classifier_activation=None,
))
cnn_model.add(keras.layers.Dense(1, activation="relu", kernel_regularizer=regularizer))

cnn_model.summary()

# Optimizer mit bestimmter Lernrate festlegen
opt = keras.optimizers.Adam(learning_rate=0.001)  # 0.01

# Loss-Funktion, Optimizer und Metrik dem neuronalen Netz zuweisen
# R2Score = keras.metrics.R2Score()
# cnn_model.compile(loss="mse", optimizer=opt, metrics=R2Score)
# ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32: <tf.Tensor 'ExpandDims_1:0' shape=(None, 1) dtype=int32>
cnn_model.compile(loss="mse", optimizer=opt, metrics=[r2_score, "accuracy"])

# learning process
history = cnn_model.fit(train_data, batch_size=32,
                        epochs=100,
                        validation_data=(test_data))


# def r2_score(y_true, y_pred):
# y_diff = tf.reduce_sum(tf.square(y_true-y_pred))
# y_square = tf.reduce_sum(tf.square(y_true-tf.reduce_mean(y_true)))
# return 1-y_diff/y_square
exit()

# model.fit(train_data , epochs=30, batch_size=40, validation_data=(test_data))

# show Figures
plt.show()

# end of file
