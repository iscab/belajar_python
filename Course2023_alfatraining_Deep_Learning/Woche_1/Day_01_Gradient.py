# Python, using Anaconda environment
# Week 1, Day 1

import numpy as np
import tensorflow as tf

z = tf.constant(3.0)
x = tf.Variable(3.0, name="x")

with tf.GradientTape() as tape:
    y = x**2
dy_dx = tape.gradient(y, x)

A = np.array([1, 2])
B = tf.Variable(A)
C = B.numpy()

print(y, type(y))
print(dy_dx, type(dy_dx))

print(A, type(A))
print(B, type(B))
print(C, type(C))

# end of file
