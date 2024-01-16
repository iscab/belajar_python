# Python, using Anaconda environment
# Week 1, Day 2

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# MLP-Classifier mit 2 Hidden-Layer mit 7, 4 Neuronen
# (7 Neuronen in der ersten Hiddenschicht, 4 Neuronen in der zweiten Hiddenschicht)
clf = MLPClassifier(hidden_layer_sizes=(7, 4), max_iter=10_000)
clf.fit(X_train, y_train)

print("Accuracy:  ", accuracy_score(y_test, clf.predict(X_test)))
print(y_test, "\n", clf.predict(X_test))
print("Parameters:  ")
print("number of layers:  ", len(clf.coefs_))
for idx in range(0, len(clf.coefs_)):
    print(clf.coefs_[idx].shape)
print("\n")

# MLP-Classifier mit 3 Hidden-Layer mit 7, 6, 4 Neuronen
# (7 Neuronen in der ersten Hiddenschicht, 6 Neuronen in der zweiten Hiddenschicht,
# und 4 Neuronen in der dritten Hiddenschicht)
clf_a = MLPClassifier(hidden_layer_sizes=(7, 6, 4), max_iter=10_000)
clf_a.fit(X_train, y_train)

print("Accuracy:  ", accuracy_score(y_test, clf_a.predict(X_test)))
print(y_test, "\n", clf_a.predict(X_test))
print("Parameters:  ")
print("number of layers:  ", len(clf_a.coefs_))
for idx in range(0, len(clf_a.coefs_)):
    print(clf_a.coefs_[idx].shape)
print("\n")


# print("Parameters 1:  \n", clf.coefs_, type(clf.coefs_), "\n")
# print("Parameters 2:  \n", clf_a.coefs_, type(clf_a.coefs_), "\n")

# MLP-Classifier mit 2 Hidden-Layer mit 7, 2 Neuronen
# (7 Neuronen in der ersten Hiddenschicht, 2 Neuronen in der zweiten Hiddenschicht)
clf_b = MLPClassifier(hidden_layer_sizes=(7, 2), max_iter=10_000)
clf_b.fit(X_train, y_train)

print("Accuracy:  ", accuracy_score(y_test, clf_b.predict(X_test)))
print(y_test, "\n", clf_b.predict(X_test))
print("Parameters:  ")
print("number of layers:  ", len(clf_b.coefs_))
for idx in range(0, len(clf_b.coefs_)):
    print(clf_b.coefs_[idx].shape)
print("\n")


# end of file
