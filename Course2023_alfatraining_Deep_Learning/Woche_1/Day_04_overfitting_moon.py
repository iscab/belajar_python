# Python, using Anaconda environment
# Week 1, Day 4
# forked from Bernd Ebenhoch, modified by adding Drop Out (and also L2 regularizers)

# Wir wollen in diesem Beispiel erforschen wie sich Overfitting vermeiden
# und dadurch die Test-Metrik verbessern lässt
# Das Beispiel ist primär so designt, dass ein starkes Overfitting vorliegt.

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow import keras


# Zufallszahlengenerator in keras initialisieren
keras.utils.set_random_seed(0)

# Datensatz erzeugen
x, y = make_moons(noise=0.5, random_state=0, n_samples=1000)

# Aufteilung in Trainings- Test- und Validierungsdaten
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=0)

x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, test_size=0.5, random_state=0)

# Daten visualisieren
plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm')
plt.title('Original Data')
# plt.show()


# Modell designen für Binärklassifikation
drop_out_rate = 0.8
model = keras.models.Sequential()
model.add(keras.layers.Dense(40, activation='relu', input_shape=[2]))
# model.add(keras.layers.Dropout(rate=drop_out_rate, seed=None))
model.add(keras.layers.Dense(80, activation='relu', kernel_regularizer=keras.regularizers.L2(0.01)))
# model.add(keras.layers.Dense(80, activation='relu'))
model.add(keras.layers.Dropout(rate=drop_out_rate))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dropout(rate=drop_out_rate))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dropout(rate=drop_out_rate))
model.add(keras.layers.Dense(80, activation='relu'))
model.add(keras.layers.Dropout(rate=drop_out_rate))
model.add(keras.layers.Dense(40, activation='relu'))
# model.add(keras.layers.Dropout(rate=drop_out_rate))
model.add(keras.layers.Dense(1, activation='sigmoid'))


# Optimizer mit bestimmter Lernrate festlegen
opt = keras.optimizers.Adam(learning_rate=0.01)  # 0.01

# Loss-Funktion, Optimizer und Metrik dem neuronalen Netz zuweisen
model.compile(loss='binary_crossentropy',
              optimizer=opt, metrics=['accuracy'])

# Eine Zusammenfassung es Modells anzeigen
model.summary()
print(x_train.shape, y_train.shape)

# das Modell trainieren
history = model.fit(x_train, y_train, batch_size=500,
                    epochs=500,
                    validation_data=(x_test, y_test))


# die Lernkurvven plotten
plt.figure()
plt.plot(history.history['accuracy'], label='Train_accuracy')
plt.plot(history.history['val_accuracy'], label='Test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()

# Eine Prognose erstellen und darstellen
y_pred = model.predict(x, verbose=False)
plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=(y_pred), cmap='coolwarm')
plt.title('Predicted Data')
# plt.show()

# Das Ergebnis beurteilen
print()
print('Evaluation from learning curves:')
print('Train_accuracy:', history.history['accuracy'][-1])
print('Test_accuracy:', history.history['val_accuracy'][-1])

print()
print('Evaluation by evaluate:')
score = model.evaluate(x_train, y_train, verbose=False)
print('Train_accuracy', score[1])

score = model.evaluate(x_test, y_test, verbose=False)
print('Test_accuracy:', score[1])


# Nach der Optimierung
print()
print('Evaluation after optimisation')
score = model.evaluate(x_val, y_val, verbose=False)
print('Val_accuracy:', score[1])

# show plots
plt.show()

# end of file
