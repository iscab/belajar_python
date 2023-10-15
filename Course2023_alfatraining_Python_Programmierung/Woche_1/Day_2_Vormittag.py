# Python, using Anaconda environment
# Week 1, Day 2
from numpy import array as np_array
import pandas

# Day 2 Folien
print("\nDay 2 lecture: ")
zahl = 0
andere_zahl = 0.0
anderes = float("5.93")
# anderes = int("5.93")  # error  ValueError: invalid literal for int() with base 10: '5.93'

print("zahl:  ", zahl, type(zahl))
print("andere_zahl:  ", andere_zahl, type(andere_zahl))
print("anderes:  ", anderes, type(anderes))
print("\n")

# Day 2: Übung
print("Day 2 Übung")

print(5 == 7)  # false
print(5 != 7)  # true

print(2 > 3)  # false
print(2 < 4)  # true
print(5 >= 5)  # true
print(5 <= 7)  # true

x=3
y=4
print("\n", x, y)
print(x == y or x < y)
print(x == y and y > 2)
print(not x == y)
print("\n")

# packages
print("numpy array: ")
xx = np_array([1.3, 3.4, 8.7])
print(xx)
print("\n")

# Modulo
print("Modulo:  ")
print("7%2 = ", 7%2)
print("7//2 = ", 7//2)
print("\n")
