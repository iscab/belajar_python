# Python, using Anaconda environment
# Week 3, Day 13

import numpy as np
import matplotlib.pyplot as plt

import helper_func.day12func as hfun

# np.random.seed(42)

# get data
dataset = np.loadtxt("Autoklassifizierung.csv", delimiter=",")

y = dataset[:, 0]
X = dataset[:, 1:3]

MyNN = hfun.MyNeuralNetwork(learning_rate=0.025, tmax=10000, skaliert=True)

# print(X[0:10, :], type(X))
X_new = MyNN.prepare_input(X)
# print(X_new[0:10, :], type(X_new))

MyNN.fit(X, y)
y_pred = MyNN.prepare_input(X)
# print(y - y_pred)
# print(y_pred)


# end of file
