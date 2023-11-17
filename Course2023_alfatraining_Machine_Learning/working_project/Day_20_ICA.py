# Python, using Anaconda environment
# Week 4, Day 20

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

random_seed = 42
np.random.seed(random_seed)

# prepare white wine data, with pandas
wine_data = pd.read_csv("winequality-white.csv", delimiter=";")
# print(wine_data, type(wine_data))
# print(wine_data.describe())

# data for ICA
y = wine_data["quality"]
X = wine_data.drop("quality", axis=1)

# Z-scale
z_scale = StandardScaler()
X_scaled = z_scale.fit_transform(X)

# print(X_scaled[:3])
# print("\n")

# ICA
# ica = FastICA(max_iter=1000, tol=1e-5,  random_state=42)
ica = FastICA(n_components=9, max_iter=1000, tol=1e-5,  random_state=42)
X_new = ica.fit_transform(X_scaled)

# independent components
print(ica.components_, "\n")
print(ica.mixing_, "\n")
print(ica.n_features_in_)
print(ica.n_components)
print(ica.mean_)
print(ica.n_iter_)
print("\n")

print(X_scaled[:3])
print(X_scaled.shape)
print(X_new[:3])
print(X_new.shape)
print("\n")

#







# end of file
