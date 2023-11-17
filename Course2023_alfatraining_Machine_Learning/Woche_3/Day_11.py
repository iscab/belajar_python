# Python, using Anaconda environment
# Week 3, Day 11
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression

clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
print(iris.keys())
print("Feature:  ", iris.feature_names)
print("Target:  ", iris.target_names)
print("\n")

n_fold = 10

clf.fit(iris.data, iris.target)
print("Decision Tree training score:  ", clf.score(iris.data, iris.target))
print(clf.feature_importances_)
print("\n")

my_score_1 = cross_val_score(clf, iris.data, iris.target, cv=n_fold)
print("Decision Tree cross-validation training score: \n", my_score_1, type(my_score_1))
print("mean:  ", my_score_1.mean())
print("standard deviation:  ", my_score_1.std())

print(clf.get_params())
# print(clf.feature_importances_())  # sklearn.exceptions.NotFittedError: This DecisionTreeClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
print("\n")


my_score_2 = cross_validate(clf, iris.data, iris.target, cv=n_fold)
# print(my_score_2, type(my_score_2))
print("Decision Tree cross-validation training score: \n", my_score_2["test_score"], type(my_score_2["test_score"]))
print("mean:  ", my_score_2["test_score"].mean())
print("standard deviation:  ", my_score_2["test_score"].std())

# print(clf.feature_importances_)  # sklearn.exceptions.NotFittedError: This DecisionTreeClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
print("\n")

log_model = LogisticRegression(random_state=0, max_iter=1000)
log_model.fit(iris.data, iris.target)
print("Logistic Regression training score: ", log_model.score(iris.data, iris.target))
print("Steigungsfaktor:  ", log_model.coef_)
print("axis:  ", log_model.intercept_)
print("\n")

my_score_3 = cross_val_score(log_model, iris.data, iris.target, cv=n_fold)
print("Logistic Regression cross validation training score: ", my_score_3)
print("mean:  ", my_score_3.mean())
print("standard deviation:  ", my_score_3.std())
print("\n")

lin_model = LinearRegression()
lin_model.fit(iris.data, iris.target)
print("Linear Regression training score: ", lin_model.score(iris.data, iris.target))
print("Steigungsfaktor:  ", lin_model.coef_)
print("axis:  ", lin_model.intercept_)
print("\n")

my_score_4 = cross_val_score(lin_model, iris.data, iris.target, cv=n_fold)
print("Linear Regression cross validation training score: ", my_score_4)
print("mean:  ", my_score_4.mean())
print("standard deviation:  ", my_score_4.std())
print("\n")

# Aufgabe 2-2
np.random.seed(42)
n_row = 1000
x = 10* np.random.rand(n_row, 2)
y = np.zeros(n_row)

index = np.flatnonzero(x[:,0] < 2)
# print(index)
# print(x[index])
y[index] = 1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("training data size:  ", X_train.shape)
print("training target size:  ", y_train.shape)
print("test data size:  ", X_test.shape)
print("test target size:  ", y_test.shape)
print("\n")

smallTree = DecisionTreeRegressor(random_state=0)
smallTree.fit(X_train, y_train)

print("Regression Tree training score:  ", smallTree.score(X_train, y_train))
print("Regression Tree test score:  ", smallTree.score(X_test, y_test))
print(smallTree.feature_importances_)
print("\n")

my_score_5 = cross_val_score(smallTree, X_train, y_train, cv=n_fold)
print("Regression Tree cross validation training score: ", my_score_5)
print("mean:  ", my_score_5.mean())
print("standard deviation:  ", my_score_5.std())
print("\n")

# end of file
