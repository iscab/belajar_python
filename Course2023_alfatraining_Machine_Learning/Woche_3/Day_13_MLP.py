# Python, using Anaconda environment
# Week 3, Day 13

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# import helper_func.day12func as hfun

np.random.seed(4)

# prepare Boston housing data
housing_data = np.loadtxt("BostonFeature.csv", delimiter=",")
housing_output = np.loadtxt("BostonTarget.csv", delimiter=",")
print("original data:  ")
print(housing_output.shape)
print(housing_data.shape)
print("\n")

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

# min max scaling
minmax = MinMaxScaler()
housing_data_minmax = minmax.fit_transform(housing_data)

# test and train data split
X_train, X_test, y_train, y_test = train_test_split(housing_data_minmax, housing_output, test_size=0.15)

print("training data: ")
print(X_train.shape)
print(y_train.shape)
print("\n")

print("test data: ")
print(X_test.shape)
print(y_test.shape)
print("\n")

# MLP classifier
nn1 = MLPRegressor(hidden_layer_sizes=(10, 3),
                   activation="tanh",  # relu, tanh
                   solver="adam",  # adam, lbfgs
                   max_iter=100_000,
                   tol=0.1)

nn1.fit(X_train, y_train)
y_pred = nn1.predict(X_test)
# print(y_pred)

housing_minmax_training_score = nn1.score(X_train, y_train)
housing_minmax_test_score = nn1.score(X_test, y_test)
print("Boston housing MLP Regressor train score:  ", housing_minmax_training_score)
print("Boston housing MLP Regressor test score:  ", housing_minmax_training_score)



# end of file
