# Python, using Anaconda environment
# Week 1, Day 3
import datetime as dt
import this as zen

# Part 1: Date and Time
print("Part 1: Date and Time \n")

# Exercise 1
print("Exercise 1: ")
zeit_jetzt = dt.datetime.now()
print("Current date and time:  ", zeit_jetzt, type(zeit_jetzt))
print("\n")

# Exercise 2
print("Exercise 2: ")
print("year:  ", zeit_jetzt.year, type(zeit_jetzt.year))
print("month:  ", zeit_jetzt.month, type(zeit_jetzt.month))
print("day:  ", zeit_jetzt.day, type(zeit_jetzt.day))
print("hour:  ", zeit_jetzt.hour, type(zeit_jetzt.hour))
print("minute:  ", zeit_jetzt.minute, type(zeit_jetzt.minute))
print("second:  ", zeit_jetzt.second, type(zeit_jetzt.second))
print("microsecond:  ", zeit_jetzt.microsecond, type(zeit_jetzt.microsecond))
print("\n")

# Part 2: String Manipulation
print("Part 2: String Manipulation \n")

# Exercise 3
print("Exercise 3: save the zen of python as a block string  \n")
# import this as zen
zen_string = "".join([zen.d.get(c, c) for c in zen.s])
print(zen_string, type(zen_string))
print("\n")

# Exercise 4
print("Exercise 4: Delete all P in the variable  \n")
zen4 = zen_string.replace("P","")  # P capital
# zen4 = zen4.replace("p","")  # p small
print(zen4, type(zen4))
print("\n")

# Exercise 5
print("Exercise 5: Split the variable at every y  ")
zen5 = zen_string.split("y")
print(zen5, type(zen5))
print("\n")

# Exercise 6
print("Exercise 6: Split the variable at every dot  ")
zen6 = zen_string.split(".")
print(zen6, type(zen6))
print("\n")

# Exercise 7
print("Exercise 7: Replace every y with a Y  \n")
zen7 = zen_string.replace("y","Y")
print(zen7, type(zen7))
print("\n")

# Exercise 8
print("Exercise 8: at which index the word complex is in the variable  ")
idx_complex = zen_string.index("complex")
print(idx_complex, type(idx_complex))
print(zen_string[idx_complex])
print(zen_string[idx_complex:idx_complex+7])
print("\n")

# Exercise 9
print("Exercise 9: how often the word \"than\" is used in the variable  ")
count_than = zen_string.count("than")
print(count_than, type(count_than))
print("\n")

# Exercise 10
print("Exercise 10: replace the dot with bracket, then number  ")
z10 = zen_string.replace(".","{}")
# print(z10, type(z10))
count_bracket = z10.count("{}")
print("number of brackets:  ", count_bracket, type(count_bracket))

list_of_idx = list(range(1,count_bracket+1))  # can be a list or a tuple
print(list_of_idx, type(list_of_idx))
z10 =z10.format(*list_of_idx)  # don't forget the star as a pointer to the values inside the list
print(z10, type(z10))
print("\n")

# Exercise 11
print("Exercise 11: how long is this variable?  ")
zen_length = len(zen_string)
print(zen_length, type(zen_length))
print("\n")

# Exercise 12
print("Exercise 12: how many whitespaces are inside this variable?  ")
count_whitespace = zen_string.count(" ")
print(count_whitespace, type(count_whitespace))
print("\n")

# Exercise 13
print("Exercise 13: How long is the variable without the whitespaces and without the dots?  ")
count_dots = zen_string.count(".")
count_nospaces_nodots = zen_length - count_whitespace - count_dots
print(count_nospaces_nodots)
print("\n")

# Exercise 14
print("Exercise 14: my name in a variable  ")
myName = "Ignatius Sapto Condro Atmawan Bisawarna"
print(myName, type(myName))
print("\n")

# Exercise 15
print("Exercise 15: delete every whitespace in the variables  ")
myName_nospaces = myName.replace(" ", "")
print(myName_nospaces, type(myName_nospaces))
print("\n")

# Exercise 16
print("Exercise 16:  ")
myName_capital = myName.upper()
myName_small = myName.lower()
print(myName_capital, type(myName_capital))
print(myName_small, type(myName_small))

myName_capital_first = myName.capitalize()
myName_capital_lasts = myName_capital_first.swapcase()
print(myName_capital_first, type(myName_capital_first))
print(myName_capital_lasts, type(myName_capital_lasts))
# OR
# myName.title()
# myName.title().swapcase()
print("\n")

# Exercise 17
print("Exercise 17: add age to name  ")
myName_with_age = myName + " 43"
print(myName_with_age, type(myName_with_age))
print("\n")

# Exercise 18
print("Exercise 18: fill the variable from the beginning with three zeros  ")
myName_with_zeros = "000"+ myName[3:]
print(myName_with_zeros, type(myName_with_zeros))
print("\n")
print("Teacher solution:  \n")
print(myName.zfill(3 + len(myName)))
print("\n")
