# Python, using Anaconda environment
# Week 2, Day 10
import numpy as np

from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

neigh = KNeighborsClassifier(n_neighbors=3)

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

neigh.fit(X, y)
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))
print("\n")
print(neigh.predict([[2.5]]))
print(neigh.predict_proba([[2.5]]))
print("\n")
print(neigh.predict([[5.5]]))
print(neigh.predict_proba([[5.5]]))
print("\n")
print(neigh.predict([[-5.5]]))
print(neigh.predict_proba([[-5.5]]))
print("\n")


clf = LocalOutlierFactor(n_neighbors=2)
X = [[-1.1], [0.2], [101.1], [0.3]]
clf.fit_predict(X)
print(clf.negative_outlier_factor_)
print("\n")


# load iris data
iris_data = load_iris(return_X_y=True)
# print(iris_data, type(iris_data))
# print(type(iris_data[0]))
# print(type(iris_data[1]))
X = iris_data[0]
y = iris_data[1]
print("data size:  ", X.shape)
print("target size:  ", y.shape)

# test and train data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

print("train data size: ", X_train.shape)
print("train target size: ", y_train.shape)
print("test data size: ", X_test.shape)
print("test target size: ", y_test.shape)
print("\n")

# neigh_2 = KNeighborsClassifier(n_neighbors=2)
# neigh_3 = KNeighborsClassifier(n_neighbors=3)
knn = [2, 3, 5, 7]
for n in knn:
    for my_weights in ["uniform", "distance"]:
        for my_metric in ["euclidean", "manhattan", "cosine"]:
            print(30 * "-")
            neigh = KNeighborsClassifier(n_neighbors=n, weights=my_weights, metric=my_metric)
            # neigh = KNeighborsClassifier(n_neighbors=n, weights=my_weights)
            print(neigh)
            print(f"{n}-nearest neighbour with {my_metric} distance:  ")
            print("the weight is ", my_weights)

            # training
            neigh.fit(X_train, y_train)

            # inference/predict
            y_predict = neigh.predict(X_test)
            y_prob = neigh.predict_proba(X_test)
            print("prediction:  ", y_predict)
            # print("probabilites:  \n", y_prob)
            print("\n")
            train_score = neigh.score(X_train, y_train)
            test_score = neigh.score(X_test, y_test)
            print("train score:  ", train_score)
            print("test score:  ", test_score)
            print("\n")

# end of file
