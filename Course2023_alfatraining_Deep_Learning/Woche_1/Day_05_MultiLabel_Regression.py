# Python, using Anaconda environment
# Week 1, Day 5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

my_random_seed = 42

# Load data
# read this:  https://stackoverflow.com/questions/27896214/reading-tab-delimited-file-with-pandas-works-on-windows-but-not-on-mac
data_df = pd.read_csv("IT_Equipment.csv", sep="\t", lineterminator="\r")
print(data_df)
print(data_df.describe())
print("\n")

# Prepare inputs and outputs
X_equipment = data_df["Budget"]
y_equipment = data_df.drop("Budget", axis="columns")
# print(X_equipment)
# print(y_equipment)
n_input = 1
n_output = y_equipment.shape[1]
print("number of inputs:  ", n_input)
print("number of outputs:  ", n_output)
print("\n")

# train and test data split
X_train, X_test, y_train, y_test = train_test_split(X_equipment, y_equipment, test_size=0.2, random_state=my_random_seed)

print(X_train.shape, type(X_train))
print(y_train.shape, type(y_train))
print(X_test.shape, type(X_train))
print(y_test.shape, type(y_train))
print("\n")

# scaling
scaler = MinMaxScaler()
# scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values.reshape(1, -1).T)
X_test_scaled = scaler.transform(X_test.values.reshape(1, -1).T)
# read this:  https://stackoverflow.com/questions/45554008/error-in-python-script-expected-2d-array-got-1d-array-instead
y_train_scaled = y_train.values.astype(float)
y_test_scaled = y_test.values.astype(float)

print(X_train_scaled.shape, type(X_train_scaled))
print(X_test_scaled.shape, type(X_test_scaled))
print(y_train_scaled.shape, type(y_train_scaled))
print(y_test_scaled.shape, type(y_test_scaled))


# Model
drop_out_rate = 0.5
regulator = tf.keras.regularizers.L2(0.01)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=[n_input], kernel_regularizer=regulator))
model.add(tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer=regulator))
model.add(tf.keras.layers.Dropout(rate=drop_out_rate))
model.add(tf.keras.layers.Dense(64, activation='tanh', kernel_regularizer=regulator))
model.add(tf.keras.layers.Dropout(rate=drop_out_rate))
model.add(tf.keras.layers.Dense(8, activation='sigmoid', kernel_regularizer=regulator))
model.add(tf.keras.layers.Dropout(rate=drop_out_rate))
model.add(tf.keras.layers.Dense(n_output, activation='relu', kernel_regularizer=regulator))

# optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.01)  # 0.01

# metric
R2Score = tf.keras.metrics.R2Score()
# we expect R2Score > 0.9

# loss function
model.compile(loss="mse", optimizer=opt, metrics=R2Score)
model.summary()

# training the model
history = model.fit(X_train_scaled, y_train_scaled, batch_size=40, epochs=500,
                    validation_data=(X_test_scaled, y_test_scaled))

# Visualization
plt.figure()
plt.plot(history.history["loss"], label="Train data")
plt.plot(history.history["val_loss"], label="Test data")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend()
# plt.show()

plt.figure()
plt.plot(history.history['r2_score'], label='Train data')
plt.plot(history.history['val_r2_score'], label='Test data')
plt.xlabel('Epoch')
plt.ylabel('r2_score')
plt.legend()
# plt.show()

# show figures
plt.show()

# end of file
