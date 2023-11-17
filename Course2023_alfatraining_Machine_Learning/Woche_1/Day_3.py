# Python, using Anaconda environment
# Week 1, Day 3

import numpy as np
A = [[25, 24, 26], [23, -2, 3], [0, 1, 2]]
B = np.array(A)
C = np.matrix("1 2 3; -1 0 1; 1 1 4")#
print(A, type(A))
print(B, type(B))
print(C, type(C))
print("\n")

# Aufgabe a
print("Aufgabe a")
A_ones = np.ones((3, 4))
print(A_ones, type(A_ones))
A_zeros = np.zeros((2, 4))
print(A_zeros, type(A_zeros))
print("\n")

A_random = np.random.rand(20, 2)  # uniform between 0 and 1
print(A_random, type(A_random))
print("\n")
A_random_u = 10*np.random.rand(20, 2)  # uniform between 0 and 10
print(A_random_u, type(A_random_u))
print("\n")
A_random_int = np.random.randint(0, 10, size=(20, 2))  # uniform integer between 0 and 10
print(A_random_int, type(A_random_int))
print("\n")
A_random_n = np.random.randn(20, 2) + 3  # normal over 1
print(A_random_n, type(A_random_n))
mean_A = np.mean(A_random_n)
print(mean_A, type(mean_A))
print("\n")
# exit()
A_random_n = np.random.standard_normal((20, 2)) # + 1  # normal oberhalb 1
print(A_random_n, type(A_random_n))
print("\n")
A_random_n = np.random.normal(3,1,(20, 2))  # normal oberhalb 1
print(A_random_n, type(A_random_n))
print("\n")
# exit()

print("Aufgabe b")
A_sample = np.random.random_sample((20, 2))
print(A_sample, type(A_sample))
print("\n")

print("Aufgabe c")
a1 = np.arange(0,12)
print(a1, type(a1))
a1_lin = np.linspace(0, 12, 12)
print(a1_lin, type(a1_lin))
print("\n")

a1_r43 = a1.reshape(4, 3)
print(a1_r43, type(a1_r43))

a1_r26 = a1.reshape(2, 6)
print(a1_r26, type(a1_r26))
a1_r34 = a1_r26.reshape(3, 4)
print(a1_r34, type(a1_r34))

# a1_r35 = a1.reshape(3, 5)  # ValueError: cannot reshape array of size 12 into shape (3,5)
# print(a1_r35, type(a1_r35))
print("\n")

# a2 = np.logspace(10,stop=100, endpoint=1000_000, num=6)
# print(a2, type(a2))



# end of file
