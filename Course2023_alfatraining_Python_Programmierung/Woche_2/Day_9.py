# Python, using Anaconda environment
# Week 2, Day 9
import random
import helper_func.day9func as hfun

x = 7
mylist = random.sample(range(-100, 100), x)
print("before sorting list 1:  ", mylist)
"""mylist2 = random.sample(range(-100, 100), x-1)
print("before sorting list 2:  ", mylist2)"""
print("\n")

# test_out = hfun.split_two(mylist)
# print(test_out)
# print("\n")

# test_out2 = hfun.split_two(mylist2)
# print(test_out2)
# print("\n")

# exit()

# hfun.split_em(mylist, mylist2)
# print("\n")
#hfun.split_em(*test_out)
#print("\n")
# hfun.split_em(*test_out2)
# print("\n")

test_out = hfun.split_em(mylist)
print("after splitting: ", test_out, type(test_out))
print("\n")

"""somelist = [4,1]
somelist = hfun.swap_two_element(somelist)
print(somelist)
somelist = [1,7]
somelist = hfun.swap_two_element(somelist)
print(somelist)"""




test_out = hfun.merge_sort(mylist)
print("after xxx: ", test_out, type(test_out))
print("\n")

