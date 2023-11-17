# Python, using Anaconda environment
# Week 3, Day 14

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# prepare Boston housing data
housing_data = np.loadtxt("BostonFeature.csv", delimiter=",")
housing_output = np.loadtxt("BostonTarget.csv", delimiter=",")
print("original data:  ")
print(housing_output.shape)
print(housing_data.shape)
print("\n")

# scaling: Z-score
scaler = StandardScaler()
scaler.fit(housing_data)
X = scaler.transform(housing_data)

# The PCA model
pca = PCA()
X_new = pca.fit_transform(X)

print("explained variance ratio:  ", pca.explained_variance_ratio_)
print("explained variance:  ", pca.explained_variance_)
print("\n")
# print("components: \n", pca.components_)
# print("components in absolute: \n ", abs(pca.components_))
# print("\n")

n_component = 0
sum_exp_var_ratio = 0.0
print(sum(pca.explained_variance_ratio_[0:9]))
print(len(pca.explained_variance_ratio_[0:9]))

chosen_variance_ratio = pca.explained_variance_ratio_[0:9]
print(chosen_variance_ratio)
print("\n")

pca_new = PCA(n_components=9)
X_trans = pca_new.fit_transform(X)

print("explained variance ratio:  ", pca_new.explained_variance_ratio_)
print("explained variance:  ", pca_new.explained_variance_)
print("\n")
# print("components: \n", pca_new.components_)
# print("components in absolute: \n ", abs(pca_new.components_))
# print("\n")

print(X[0:5, :], "\n")
print(X_new[0:5, :], "\n")
print(X_trans[0:5, :], "\n")


# end of file
