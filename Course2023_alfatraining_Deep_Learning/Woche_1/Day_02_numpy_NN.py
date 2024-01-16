# Python, using Anaconda environment
# Week 1, Day 2
# forked from Bernd Ebenhoch, modified by adding bias in neural network

# In diesem Beispiel erstellen wir ein eigenes neuronales Netz
# nur mit Hilfe von NumPy-Funktionen und fitten es auf den Iris-Datensatz


from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Iris-Datensatz laden
iris = load_iris()
x = iris.data

y = iris.target.reshape(-1, 1)

print(y)


# Labels für Klassifikation onehotencoden
enc = OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()

print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Dimensionen definieren
D_in = 4
H = 100  # Eine einzelne Hidden-Schicht mit 100 Neuronen
D_out = 3  # 3 Neuronen in der Ausgabeschicht für die 3 Klassen


# Gewichtungen mit Zufallszahlen initialisieren
w1 = np.random.randn(D_in, H)  # Merkmale --> Hiddenschicht
w2 = np.random.randn(H, D_out)  # Hiddenschicht --> Output

# biases
b1 = np.random.randn(1, H)
b2 = np.random.randn(1, D_out)


# Leere Liste um die Loss-Werte zu speichern
losses = []

# Lernrate definieren
learning_rate = 1e-4  # 0.0001
n = 120  # Anzahl der Trainingsdatenpunkte


# Das neuronale Netz über 10000 Epochen trainieren
nEpoch = 10_000
for t in range(nEpoch):

    # Neuronales Netz in Vorwärtsrichtung durchlaufen
    y1 = x_train.dot(w1) + b1
    # b1_x = np.ones((len(x_train), 1)).dot(b1)
    # y1 = y1 + b1_x
    y1_relu = np.maximum(y1, 0)

    y_pred = y1_relu.dot(w2) + b2   # Am Output haben wir keine Aktivierungsfunktion
    # print(len())
    # b2_y = np.tile(b2, (y1_relu.shape[0], 1))
    # y_pred = y_pred + b2_y

    # Loss berechnen
    loss = np.square(y_pred - y_train).sum()/n
    print(t, loss)
    losses.append(loss)

    # Backpropagation mit Gradientenabstieg
    grad_y_pred = 2.0 * (y_pred - y_train)/n
    dloss_dw2 = y1_relu.T.dot(grad_y_pred)  # .T macht eine Transponierung
    dloss_db2 = np.sum(grad_y_pred, axis=0)

    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h_relu[y1 < 0] = 0
    dloss_dw1 = x_train.T.dot(grad_h_relu)
    dloss_db1 = np.sum(grad_h_relu, axis=0)

    # Gewichtungen aktualisieren
    w1 = w1 - learning_rate * dloss_dw1
    w2 = w2 - learning_rate * dloss_dw2
    b1 = b1 - learning_rate * dloss_db1
    b2 = b2 - learning_rate * dloss_db2


# Neuronales Netz verwenden
y1 = x_test.dot(w1)
y1_relu = np.maximum(y1, 0)
y_pred = y1_relu.dot(w2)


for i in range(len(y_test)):
    print(i+1, "Y:", (y_test[i]), "| Y-Pred:", (y_pred[i]))

print()
print()


for i in range(len(y_test)):
    print(i+1, "Y:", np.argmax(y_test[i]), "| Y-Pred:", np.argmax(y_pred[i]))

y_test = np.argmax(y_test, axis=1)  # Macht das OneHotEncoding rückgängig
y_pred = np.argmax(y_pred, axis=1)

print('Test Accuracy', accuracy_score(y_test, y_pred))

# Loss-Werte als Funktion der Epochen grafisch darstellen
plt.plot(range(nEpoch), losses)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.show()


# end of file
