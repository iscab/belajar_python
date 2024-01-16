# Python, using Anaconda environment
# Week 1, Day 1

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


x = tf.Variable(tf.range(0, 4 * np.pi, delta=0.01))
# f = tf.math.sin(x)

with tf.GradientTape() as tape:
    f = tf.math.sin(x)

df_dx = tape.gradient(f, x)

plt.plot(x, f, color="blue")
plt.plot(x, df_dx, color="red")
plt.show()


# end of file
