# Python, using Anaconda environment
# Week 2, Day 8
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Bayesian

rng = np.random.RandomState(1)
print(rng, type(rng))

X = rng.randint(5, size=(6, 100))
# print(X, type(X))
Y = np.array([1, 2, 3, 4, 4, 5])
print(Y, type(Y))
clf = BernoulliNB(force_alpha=True)
clf.fit(X, Y)
print(X[2:3])
print(clf.predict(X[2:3]))
print(clf.get_params())
print("\n")


# Linear Regression
print("Linear Regression:  ")
model = LinearRegression(fit_intercept=True)

"""X = np.array([1, 2, 3, 4, 5])
print(X)
print("\n")
# try numpy newaxis
X = X[:, None]  # only works for numpy array, not ordinary list
print(X)
# exit()"""

# Multi
print("Multi:  ")
X = 10 * rng.rand(10, 3)
print(X, type(X))
y = 0.5 + np.dot(X, [1.5, -2., 1.])
print(y, type(y))
print("\n")

model.fit(X, y)
print("Model intercept:  ", model.intercept_, type(model.intercept_))
print("Model gradient:  ", model.coef_, type(model.coef_))
print("\n")

print("Single:  ")
# train data
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
# test data
xfit = np.linspace(0, 10, 1000)
# yfit will be output of f(xfit)
print(type(x), type(y), type(xfit))
plt.figure()
plt.scatter(x, y)
# plt.show()

model.fit(x[:, np.newaxis], y)
print("Model intercept:  ", model.intercept_, type(model.intercept_))
print("Model gradient:  ", model.coef_, type(model.coef_))
print("\n")

yfit = model.predict(xfit[:, np.newaxis])

plt.figure()
plt.scatter(x, y)
plt.plot(xfit, yfit)
# plt.show()
print("\n")

# Polynomial
print("Polynomial  ")

x = np.array([2, 3, 5, 7])
poly = PolynomialFeatures(3, include_bias=False)
Y = poly.fit_transform(x[:, None])
print(Y, type(Y))
print("\n")

# Without Pipeline
print("Polynomial without Pipeline")

poly_alone = PolynomialFeatures(7)
# linear regression model is above
# model = LinearRegression(fit_intercept=True)

# test and train data
x = 10 * rng.rand(50)  # train data
xfit = np.linspace(0, 10, 1000)  # test data

# output
y = np.sin(x) + 0.1 * rng.randn(50)  # train data
yfit = np.sin(xfit) + 0.1 * rng.randn(1000)  # test data

XX = poly_alone.fit_transform(x[:, None])
# print(XX)
print("Size:  ", XX.shape, type(XX))

# training
model.fit(XX, y)
print("Model intercept:  ", model.intercept_, type(model.intercept_))
print("Model gradient:  ", model.coef_, type(model.coef_))
print("\n")

# testing
XXfit = poly_alone.fit_transform(xfit[:, None])
yfit_alone = model.predict(XXfit)
# print(model.get_params())

plt.figure()
plt.scatter(x, y)
plt.plot(xfit, yfit_alone)
# plt.show()
print("\n")

print("Test score:  ", model.score(XXfit, yfit))

polynom_potenz = [2, 3, 5, 7, 11, 13, 17, 19]
for potenz in polynom_potenz:
    # print(potenz)
    poly_try = PolynomialFeatures(potenz)

    # training
    XX = poly_try.fit_transform(x[:, None])
    model.fit(XX, y)

    # testing
    XXfit = poly_try.fit_transform(xfit[:, None])
    yfit_out = model.predict(XXfit)

    train_score = model.score(XX, y)
    test_score = model.score(XXfit, yfit)

    print(20*"-")
    print(f"Polynom of the order {potenz}:  ")
    print("Model intercept:  ", model.intercept_, type(model.intercept_))
    print("Model gradient:  ", model.coef_, type(model.coef_))
    print("Train score:  ", train_score)
    print("Test score:  ", test_score)
    print("\n")

    plt.figure()
    plt.scatter(x, y)
    plt.plot(xfit, yfit_out)
    plt.title(f"Polynom of the order {potenz}")

# plt.show()
print("\n")


# Ridge and Lasso (L2 and L1)
housing_data = np.loadtxt("BostonFeature.csv", delimiter=",")
housing_output = np.loadtxt("BostonTarget.csv", delimiter=",")
print("original data:  ")
print(housing_output.shape)
print(housing_data.shape)
print("\n")

# test and train data split
X_train, X_test, y_train, y_test = train_test_split(housing_data, housing_output, test_size=0.15)

print("training data: ")
print(X_train.shape)
print(y_train.shape)
print("\n")

print("test data: ")
print(X_test.shape)
print(y_test.shape)
print("\n")

# linear model
L2_model = Ridge(alpha=2.8)  # Ridge
L1_model = Lasso(alpha=2.8)  # Lasso

# learning
model.fit(X_train, y_train)
L2_model.fit(X_train, y_train)
L1_model.fit(X_train, y_train)

# predict
lin_train_score = model.score(X_train, y_train)
lin_test_score = model.score(X_test, y_test)
L2_train_score = L2_model.score(X_train, y_train)
L2_test_score = L2_model.score(X_test, y_test)
L1_train_score = L1_model.score(X_train, y_train)
L1_test_score = L1_model.score(X_test, y_test)

print("Linear Regression train score:  ", lin_train_score)
print("Linear Regression test score:  ", lin_test_score)
print("Model gradient:  ", model.coef_, type(model.coef_))
print(f"max Gradient:  {np.max(np.abs(model.coef_))}   Spalte:  {np.argmax(np.abs(model.coef_))}")
print("Model intercept:  ", model.intercept_, type(model.intercept_))
print("---")
print("Ridge Regression train score:  ", L2_train_score)
print("Ridge Regression test score:  ", L2_test_score)
print("Model gradient:  ", L2_model.coef_, type(model.coef_))
print(f"max Gradient:  {np.max(np.abs(L2_model.coef_))}   Spalte:  {np.argmax(np.abs(L2_model.coef_))}")
print("Model intercept:  ", L2_model.intercept_, type(model.intercept_))
print("---")
print("Lasso Regression train score:  ", L1_train_score)
print("Lasso Regression test score:  ", L1_test_score)
print("Model gradient:  ", L1_model.coef_, type(model.coef_))
print(f"max Gradient:  {np.max(np.abs(L1_model.coef_))}   Spalte:  {np.argmax(np.abs(L1_model.coef_))}")
print("Model intercept:  ", L1_model.intercept_, type(model.intercept_))
print("\n")









# end of file
