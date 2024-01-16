# Python, using Anaconda environment
# Week 1, Day 2

import numpy as np

X = np.ones(120).reshape(-1, 12)
print(X, type(X))

# X = np.ones(120).reshape(-1, 13)  # ValueError: cannot reshape array of size 120 into shape (13)
X = np.ones(120).reshape(-1, 10)
print(X, type(X))
print("\n")

X1 = np.ones(100).reshape(10, 10)
print(X1, type(X1), "\n")

X2 = np.zeros(20).reshape(2, 10)
print(X2, type(X2), "\n")

X_conc = np.concatenate((X1, X2), axis=0)
print(X_conc, type(X_conc), "\n")

# w = np.array([[1, 2], [3, 4]])
w = np.array([[1, 2], [3, 4], [5, 6]])
x = np.array([[5], [6]])
b = np.array([[0],[1],[3]])

print("w:  \n", w, "\n",  w.shape)
print("x:  \n", x, "\n", x.shape)

# y = w * x
y = w.dot(x)
print("y:  \n", y, "\n", y.shape)

y = w @ x
print("y:  \n", y, "\n", y.shape)

y = np.dot(w, x)
print("y:  \n", y, "\n", y.shape)

y = np.dot(w, x) + b
print("y:  \n", y, "\n", y.shape)

y = w.dot(x) + b
print("y:  \n", y, "\n", y.shape)

print("\n")
y = np.sin(w.dot(x) + b)
print("y:  \n", y, "\n", y.shape)


# end of file
