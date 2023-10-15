# Python, using Anaconda environment
# Week 2, Day 8
import helper_func.day8func as hfun


# Exercises
mylist = [x for x in range(8, 20)]
results = map(hfun.sum_till, mylist)
print(list(results))
print("\n")

# quadrat labelling
print("Exercise 1 from Day 7:  ")
print("input:  ", mylist)
results = map(lambda x_text, y_num: {x_text: y_num**2}, "quadrat", mylist)  # not beautiful
print("output:  ", list(results))
print("input:  ", mylist)
results = map(lambda x_text, y_num: {x_text: y_num**2}, "quadratic equation result", mylist)  # not beautiful
print("output:  ", list(results))

