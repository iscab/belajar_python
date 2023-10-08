# Python, using Anaconda environment
# Week 3, Day 11
from functools import reduce

# read this:  https://en.wikipedia.org/wiki/Lazy_evaluation
# read this:  https://de.wikipedia.org/wiki/Lazy_Evaluation

mystring = ["Hallo", "Hallihallo", "Yo", "Lölle"]

filtered_string = filter(lambda s: s[-1] == "o", mystring)
print(mystring, type(mystring))
print(filtered_string, type(filtered_string))
print(list(filtered_string))
print("\n")

filtered_string2 = map(lambda s: s[-1] == "o", mystring)
print(mystring, type(mystring))
print(filtered_string2, type(filtered_string2))
print(list(filtered_string2))
print("\n")


# read this:  https://www.geeksforgeeks.org/reduce-in-python/

# from functools import reduce
z = reduce(lambda x, y: x + y, [1, 5, 7, 3])
print(z, type(z))
print("\n")

# from functools import reduce
z = reduce(lambda x, y: x + y, ["Hey", "Muh", "Buh"])
print(z, type(z))
print("\n")


def multipl(x, y):
    return x * y


# from functools import reduce
list_of_number = [2, 3, 5, 7]
myfun = reduce(lambda x, y: x * y, list_of_number)
print(myfun, type(myfun))
myfun = reduce(multipl, list_of_number)
print(myfun, type(myfun))
print("\n")

print(list("Hallo"))
print("\n")

mystring = input("Please input a number:  ")
try:
    mynum = int(mystring)
    print(mystring, type(mystring))
    print(mynum, type(mynum))
except:
    print("LOL, I cannot turn your input into integer")

print("\n")

ticker = True
t = 1
while ticker:
    mystring = input("Please input a number:  ")
    try:
        mynum = int(mystring)
        print(mystring, type(mystring))
        print(mynum, type(mynum))
        ticker = False
    except:
        print("LOL, I cannot turn your input into integer")
        t += 1
    if t > 5:
        ticker = False

print("Nach der Schleife  \n")

# myfun1 = "Hello"
try:
    print(myfun1)
except:
    print("Something went wrong")
else:
# finally:
    print("Everything works fine.")

print("\n")

my_dict = {"Name": "Jojon", "Age": 37}

try:
    print(myfun)
    # print(myfun1)  # NameError
    # mylist = list(myfun)  # TypeError
    # print(list_of_number[37])  # IndexError
    # print(my_dict["Test"])  # KeyError
except NameError:
    print("your name is wrong")
except TypeError:
    print("your type is wrong")
except IndexError:
    print("indexing problem")
except KeyError:
    print("Lupa kunci, ya?")
except:
    print("other error")

print("\n")


