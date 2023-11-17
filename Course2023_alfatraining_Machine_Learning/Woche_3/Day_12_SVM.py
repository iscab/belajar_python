# Python, using Anaconda environment
# Week 3, Day 12

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, load_iris
from sklearn.model_selection import GridSearchCV, train_test_split



"""x = np.linspace(-1,5, 10)
y = np.linspace(-1,5, 10)
# print(x)
Y, X = np.meshgrid(y, x)
print(X)
print(Y)"""

X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plt.show()

# Try GridSearch
iris = load_iris()
print(iris.keys())
print("Feature:  ", iris.feature_names)
print("Target:  ", iris.target_names)
print("\n")

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
print("training data size:  ", X_train.shape)
print("training target size:  ", y_train.shape)
print("test data size:  ", X_test.shape)
print("test target size:  ", y_test.shape)
print("\n")

svmc = SVC()
# parameters = {'kernel':('linear', 'rbf', 'poly'), 'C': [1, 10], 'gamma':['auto', 'scale']}
"""parameters = [{"kernel": ("linear"), "C": [1, 10]},
              {"kernel": ("poly"), "C": [1, 10], "degree": [2, 3, 5, 7]},
              {"kernel": ("rbf"), "C": [1, 10], "gamma": ["auto", "scale"]}]"""
parameters = [
    {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
    {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
     'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    {'kernel': ['poly'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
     "degree": [2, 3, 5, 7]}]


clf = GridSearchCV(svmc, parameters)
fitting = clf.fit(X_train, y_train)
print(fitting)
print(clf.cv_results_)
print("\n")

best_clf = clf.best_estimator_
print(best_clf)

y_pred = best_clf.predict(X_test)
# print(y_pred - y_test)

best_training_score = best_clf.score(X_train, y_train)
best_test_score = best_clf.score(X_test, y_test)
print("best training score:  ", best_training_score)
print("best test score:  ", best_test_score)

plt.figure()
plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train)
plt.show()

# end of file
