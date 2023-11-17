# Python, using Anaconda environment
# Week 3, Day 12

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import helper_func.day12func as hfun

np.random.seed(42)

# get data
dataset = np.loadtxt("Autoklassifizierung.csv", delimiter=",")

y = dataset[:, 0]
x = np.ones((len(y), 3))
x[:,0:2] = dataset[:, 1:3]

# scaling
minmax = MinMaxScaler()
minmax.fit(x[:,0:2])
# print(x[1:10,:], type(x))
x[:, 0:2] = minmax.transform(x[:,0:2])
# print(x[1:10, :], type(x))

# xMin = minmax.min_
xMin = minmax.data_min_
xMax = minmax.data_max_
print(f"min: {xMin} and max: {xMax}")
print("\n")

# for neural network class
eta = 0.25  # learning rate
Dw = np.zeros(3)  # weight change
w = np.random.rand(3) - 0.5  # weight
convergenz = 1
t = 0
tmax = 1000

# the fitting/learning in neural network class
print("start weights:  ", w)
print("convergenz:  ", convergenz)
while (convergenz > 0) and (t < tmax):
    t += 1
    idx_chosen = np.random.randint(len(y))
    x_chosen = x[idx_chosen, :].T
    y_chosen = y[idx_chosen]

    # error calculation
    neuro_error = y_chosen - hfun.myHeaviside(w @ x_chosen)

    # weight tuning
    for j in range(len(x_chosen)):
        Dw[j] = eta * neuro_error * x_chosen[j]
        w[j] = w[j] + Dw[j]
    convergenz = np.linalg.norm(y-hfun.myHeaviside(w @ x.T))

print("weights after:  ", w)
print("convergenz:  ", convergenz, "  after t = ", t)

y_pred = hfun.predict(x, w)
print(y - y_pred)



# end of file
