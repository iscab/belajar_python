# Python, using Anaconda environment
# Week 1, Day 1

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)

t = np.arange(0.0, 2.0, 0.01)
A = 1 + np.sin(2 * np.pi * t)
plt.plot(t, A, color="red")
plt.show()

# end of file
