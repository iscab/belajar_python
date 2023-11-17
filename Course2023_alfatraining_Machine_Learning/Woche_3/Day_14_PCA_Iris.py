# Python, using Anaconda environment
# Week 3, Day 14

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from biplot import biplot

plt.style.use('ggplot')

# Load the data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Z-score the features
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# The PCA model
pca = PCA(n_components=2)  # estimate only 2 PCs
X_new = pca.fit_transform(X)  # project the original data into the PCA space

print("explained variance ratio:  ", pca.explained_variance_ratio_)
print("explained variance:  ", pca.explained_variance_)
print("\n")
print("components: \n", pca.components_)
print("components in absolute: \n ", abs(pca.components_))
print("\n")

# the covariance matrix of the reduced space
cov_of_reduced_matrix = np.cov(X_new.T)
print("covariance matrix of the reduced space: \n", cov_of_reduced_matrix)
print("\n")

# plotting
fig, axes = plt.subplots(1,2)
axes[0].scatter(X[:,0], X[:,1], c=y)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')
axes[1].scatter(X_new[:,0], X_new[:,1], c=y)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
# plt.show()

# Die biplot-Funktion wird für die beiden wichtigsten Principal Components aufgerufen.
# biplot(X_new, pca.components_[0:2, :].T, y)
biplot(X_new, pca.components_.T, y)
plt.show()

c
# end of file
