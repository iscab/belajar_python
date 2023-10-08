# Python, using Anaconda environment
# Week 2, Day 7
import helper_func.day7func as hfun

# Exercise 1 oldies
print("Exercise 1 oldies: calculates the sum of 0 to X  \n")
test_out = hfun.sum_till(15)
print(test_out, type(test_out))
print("\n")

# Exercise 2 oldies
print("Exercise 2 oldies: calculates the area of a square, given the edge length  \n")
theArea = hfun.area_of_square(5.3)
print(theArea, type(theArea))
print("\n")

# Exercise 1
print("Exercise 1: calculate square of number with message  \n")
myword = "Banana"
mynum = 7
test_out = hfun.quadrat_label(myword, mynum)
print(test_out, type(test_out))
result_out, result_dict = hfun.quadrat_label(myword, mynum)
print(result_out, type(result_out))
print(result_dict, type(result_dict))
print("\n")

# Exercise 2
print(" Exercise 2: factorial")
x = 5
test_out = hfun.factorial_of(x)
print(test_out, type(test_out))
print("\n")

# Exercise 3
print("Exercise 3:  \n")
# TODO: just do it

# Exercise 4
print("Exercise 4:  \n")
# TODO: just do it

# Exercise 5
print("Exercise 5: separation of odd and even numbers from lists  \n")
mylist = [y for y in range(7)]
mylist2 = [y for y in range(5, 12)]
test_out = hfun.odd_or_even(mylist, mylist2)
print(test_out, type(test_out), "\n")
test_out = hfun.odd_or_even(mylist)
print(test_out, type(test_out), "\n")
print("\n")

# Exercise 6
print("Exercise 6: character count  \n")
str_text = "Kemarin aku bermimpi, bertemu dengan diri-Nya"
test_out = hfun.character_count(str_text)
print(test_out, type(test_out), "\n")
print("\n")

# Exercise 7
print("Exercise 7: sorting")
mylist = [8, 2, 6, 1, 7]
# TODO: the function
print(mylist)
# mylist.sort()  # up
mylist.sort(reverse=True)  # down
print(mylist)
print("\n")

# Exercise 8
print("Exercise 8: volumes  \n")

print("cone volume:  ")
test_out = hfun.cone_volume(2,9)
print(test_out, type(test_out))
test_out = hfun.shape_volume(hfun.cone_volume, 2, 9)
print(test_out, type(test_out))
print("\n")

print("sphere volume:  ")
test_out = hfun.sphere_volume(2)
print(test_out, type(test_out))
test_out = hfun.shape_volume(hfun.sphere_volume, 2)
print(test_out, type(test_out))
print("\n")

print("cuboid volume:  ")
test_out = hfun.cuboid_volume(2, 3, 5)
print(test_out, type(test_out))
test_out = hfun.shape_volume(hfun.cuboid_volume, 2, 3, 5)
print(test_out, type(test_out))
print("\n")

print("cylinder volume:  ")
test_out = hfun.cylinder_volume(2,9)
print(test_out, type(test_out))
test_out = hfun.shape_volume(hfun.cylinder_volume, 2, 9)
print(test_out, type(test_out))
print("\n")

# Exercise 9
print("Exercise 9: *args and **kwargs  \n")
# TODO: just do it
# read this: https://realpython.com/python-kwargs-and-args/

