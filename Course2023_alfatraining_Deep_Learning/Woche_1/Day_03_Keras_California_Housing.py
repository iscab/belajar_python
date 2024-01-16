# Python, using Anaconda environment
# Week 1, Day 3

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

# housing_data = tf.keras()
housing_data = fetch_california_housing(data_home=None, download_if_missing=True, return_X_y=False, as_frame=False)
print(type(housing_data))
print(housing_data.keys())
print(housing_data["feature_names"])
print(housing_data["target_names"])

# Wir extrahieren die Merkmale und Labels
X = housing_data.data
y = housing_data.target
# print(type(X), type(y))

# Data size
print("shape of X:  ", X.shape)
print("shape of y:  ", y.shape)

# Data visualization (better in Spyder or Jupyter Notebook, than in PyCharm)
"""for idx in range(8):
    plt.figure()
    plt.scatter(X[:, idx], y )
    plt.xlabel(housing_data.feature_names[idx], fontsize=8)
    plt.ylabel("Price")
plt.show()"""

# train and test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Merkmale normalisieren / skalieren
# Warum?
# Wenn verschiedene Merkmale unterschiedliche Zahlenbereiche aufweisen
# Da sonst das Merkmal mit den größeren Werten, die Werte der Neuronen bestimmen würden
# Wenn allgemein sehr große Zahlen vorkommen, da sonst das prognosierte Label
# auch größe Werte annimt und dadurch einen großen Loss-Wert
# und große Gradienten ergibt was das raining unstabil macht

scaler = MinMaxScaler()
# scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""for idx in range(8):
    plt.figure()
    plt.scatter(X_train_scaled[:, idx], y_train)
    plt.xlabel(housing_data.feature_names[idx], fontsize=8)
    plt.ylabel("Price")
plt.show()"""

# Das Modell erzeugen
H1 = 64  # 32
H2 = 16  # 16
inputs = keras.layers.Input(shape=X.shape[1:])
hidden1 = keras.layers.Dense(H1, activation="relu")(inputs)
hidden2 = keras.layers.Dense(H2, activation="tanh")(hidden1)
output = keras.layers.Dense(1,  activation="relu")(hidden2)

nn_model = keras.models.Model(inputs=[inputs], outputs=[output])
nn_model.summary()

R2Score = keras.metrics.R2Score()
nn_model.compile(loss="mse", optimizer="SGD", metrics=R2Score)
# nn_model.fit(X_train_scaled, y_train, epochs=100)
# nn_model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test))
nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=100, validation_data=(X_test_scaled, y_test))


# Direkt nach dem Training können wir die Lernkurven auus der historie abholen
history = nn_model.history.history

# Das history wird als Dictionary ausgegeben. wir gene uns die Keys aus
print("History keys:  ", history.keys())

# Visualization
plt.figure()
plt.plot(history["loss"], label="Train data")
plt.plot(history["val_loss"], label="Test data")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend()
# plt.show()

plt.figure()
plt.plot(history['r2_score'], label='Train data')
plt.plot(history['val_r2_score'], label='Test data')
plt.xlabel('Epoch')
plt.ylabel('r2_score')
plt.legend()
# plt.show()


# Ergebnis darstellen und beurteilen

# Prediction der Testdaten durchführen
y_pred = nn_model.predict(X_test_scaled)

# Für eine quantitative Beurteilung nutzen wir bei der Regression gerne das Bestimmtheitmaß (r2_score)
test_metric = r2_score(y_test, y_pred)

y_sim = nn_model.predict(X_train_scaled)
train_metric = r2_score(y_train, y_sim)

# Wir stellen die Prediction als Funktion von y_test grafisch
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.2)
plt.xlabel("Real Price")
plt.ylabel("Predicted Prize")
plt.plot([0, 5], [0, 5], color="black", label="ideal")
plt.title(f"Test r2_score:  {np.round(test_metric, 3)}  and Train r2_score:  {np.round(train_metric, 3)}")
plt.legend()
# plt.show()

# Feature importance
r = permutation_importance(nn_model, X_test_scaled, y_test, n_repeats=15,
                           random_state=41, scoring="neg_mean_squared_error")

for idx in range(8):
    print(housing_data.feature_names[idx])
    print(np.round(r["importances_mean"][idx], 3))
    print()

plt.show()

# end of file
