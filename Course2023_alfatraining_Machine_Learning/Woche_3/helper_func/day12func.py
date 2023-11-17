# Python, using Anaconda environment
# Week 3, Day 12

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

def myHeaviside(x):
    y = np.ones_like(x, dtype=np.float64)
    y[x <= 0] = 0
    return y

def predict(x, w):
    x_pred = np.ones((x.shape[0], 3))
    x_pred[:, 0:2] = x[:, 0:2]
    y_pred = w @ x_pred.T

    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0

    return y_pred

class MyNeuralNetwork():
    def __init__(self, learning_rate=0.25, skaliert=True, scale=MinMaxScaler, tmax=10000):
        self.n_data = 0
        self.n_var = 0
        self.eta = learning_rate
        self.weight = None
        self.delta_weight = None
        self.skaliert = skaliert
        self.scale = scale()
        self.convergenz = 1
        # t = 0
        self.tmax = tmax

    def fit(self, X, y):
        y = np.array(y[:,None])
        X_nn = self.prepare_input(X)

        # prepare weights
        self.weight = np.random.rand(int(self.n_var + 1)) - 0.5
        self.delta_weight = np.zeros(int(self.n_var + 1))

        # the fitting/learning in neural network class
        self.convergenz = 1
        t = 0
        print("start weights:  ", self.weight)
        print("convergenz:  ", self.convergenz)
        while (self.convergenz > 0) and (t < self.tmax):
            t += 1
            idx_chosen = np.random.randint(self.n_data)
            x_chosen = X_nn[idx_chosen, :].T
            y_chosen = y[idx_chosen]

            # error calculation
            neuro_error = y_chosen - myHeaviside(self.weight @ x_chosen)
            # if t < 100:
            #     print(idx_chosen, neuro_error)

            # weight tuning
            for j in range(len(x_chosen)):
                self.delta_weight[j] = self.eta * neuro_error * x_chosen[j]
                self.weight[j] = self.weight[j] + self.delta_weight[j]
                if t < 10:
                    print(self.weight, self.delta_weight)
        self.convergenz = np.linalg.norm(y - myHeaviside(self.weight @ X_nn.T))

        print("weights after:  ", self.weight)
        print("convergenz:  ", self.convergenz, "  after t = ", t)

        return self

    def predict(self, X):
        X_nn = self.prepare_input(X)
        y_pred = self.weight @ X_nn.T

        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0

        return y_pred

    def prepare_input(self, X):
        X = np.array(X)
        self.n_data = X.shape[0]
        self.n_var = X.shape[1]
        # print(self.n_var)
        # x[:,0:2] = dataset[:, 1:3]
        X_bias = np.ones((self.n_data, self.n_var + 1))
        X_bias[:, 0:self.n_var] = X[:, 1:(self.n_var + 1)]
        # print(X.shape, X_bias.shape)

        if self.skaliert:
            # x[:, 0:2] = minmax.transform(x[:,0:2])
            X_bias[:, 0:self.n_var] = self.scale.fit_transform(X_bias[:, 0:self.n_var])

        return X_bias






# end of file
