# Python, using Anaconda environment
# Week 1, Day 2
import math  # exercise 5
import numpy as np  # exercise 9

# Part 1: Data Types
print("Part 1: Data Types \n")

# Exercise 1
print("Exercise 1: ")
x1 = 2/4
x1_int = int(x1)
x1_float = float(x1)

print("x1:  ", x1, type(x1))
print("x1_int:  ", x1_int, type(x1_int))
print("x1_float:  ", x1_float, type(x1_float))
print("\n")

# Exercise 2
print("Exercise 2: ")
x2 = 1000000
x2_unterstrich = 1000_000
x2_exp = 1e6

print("x2 = 1000000:  ", x2, type(x2))
print("x2_unterstrich = 1000_000:  ", x2_unterstrich, type(x2_unterstrich))
print("x2_exp = 1e6:  ", x2_exp, type(x2_exp))
print("\n")

# Part 2: Boolean Expressions
print("Part 2: Boolean Expressions \n")

# Exercise 3
print("Exercise 3: ")
print(3 > 2, "true")  # true
print(5 != 6, "true")  # true
print(5 == 2, "false")  # false

print(3 < 2 or 2 > 1, "true")  # true

print(5 != 9 and 7 < 10, "true")  # true
print(not 5 != 9 and not 7 < 10, "false")  # false

print((not 5 != 9 and 7 < 10) or 2 * 3 > 7, "false")  # false
print(not(5 != 9 and 7 < 10) or 2 * 3 > 7, "false")  # false
print("\n")

# Part 3: Importing and Using Packages
print("Part 3: Importing and Using Packages \n")

# Exercise 4:
print("Exercise 4: \n")
print("How to install pingouin: \n    conda install -c conda-forge pingouin  \n")
print("How to install matplotlib: \n    conda install -c conda-forge matplotlib  \n")
print("How to install scipy: \n    conda install -c anaconda scipy  \n")
print("\n")

# Exercise 5:
print("Exercise 5: ")
x5 = math.sqrt(9)
print("x5:  ", x5, type(x5))
print("\n")

# Exercise 6
print("Exercise 6: ")
x6_max_inf = math.inf
x6_min_inf = -math.inf

print("x6_max_inf:  ", x6_max_inf, type(x6_max_inf))
print("x6_min_inf:  ", x6_min_inf, type(x6_min_inf))
print("\n")

# Exercise 7
print("Exercise 7: ")
print("pi:  ", math.pi, type(math.pi))
print("e:  ", math.e, type(math.e))

x7 = math.pi * math.e
print("x7:  ", x7, type(x7))
print("\n")

# Exercise 8
print("Exercise 8: ")
x8_log10 = math.log10(3)
x8_log10_also = math.log(3, 10)
x8_ln = math.log(3)
x8_ln_also = math.log(3, math.e)  # or use this for explicit base

print("log10(3) = ", x8_log10, type(x8_log10))
print("log10(3) = ", x8_log10_also, type(x8_log10_also))
print("ln(3) = ", x8_ln, type(x8_ln))
print("ln(3) = ", x8_ln_also, type(x8_ln_also))
print("\n")

# Exercise 9
print("Exercise 9: ")
mylist = [1, 2, 3, 4]
x9 = np.array(mylist)

print("mylist: ", mylist, type(mylist))
print("mylist in numpy: ", x9, type(x9))
print("\n")

x9_mean = np.mean(mylist)
x9_sum = np.sum(mylist)
x9_pop_stddev = np.std(mylist)
x9_smpl_stddev = np.std(mylist, ddof=1)

print("mean:  ", x9_mean, type(x9_mean))
print("sum:  ", x9_sum, type(x9_sum))
print("population standard deviation:  ", x9_pop_stddev, type(x9_pop_stddev))
print("sample standard deviation:  ", x9_smpl_stddev, type(x9_smpl_stddev))
print("\n")
