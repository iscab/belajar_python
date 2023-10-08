# Python, using Anaconda environment
# Week 2, Day 8
import datetime as dt
import time
import math

import Tag7_Loesungen as t7L

# reference
x = 300
print(x, " : ", id(x), type(id(x)))

x = 400
print(x, " : ", id(x), type(id(x)))

x = 5000_000_000
print(x, " : ", id(x), type(id(x)))

x1 = 300
print(x1, " : ", id(x1), type(id(x1)))

x1 = x
print(x1, " : ", id(x1), type(id(x1)))

print("\n")

y = "my string"
print(y, " : ", id(y), type(id(y)))
"""print(y[0], " : ", id(y[0]), type(id(y[0])))
print(y[1], " : ", id(y[1]), type(id(y[1])))
print(y[2], " : ", id(y[2]), type(id(y[2])))"""
for ix in y:
    print(ix, " : ", id(ix), type(id(ix)))
print("\n")

z = "the string theory"
print(z, " : ", id(z), type(id(z)))
for ix in z:
    print(ix, " : ", id(ix), type(id(ix)))
print("\n")


# list
L = [2, 3, "c"]
print(L, " : ", id(L), type(id(L)))
for ix in L:
    print(ix, " : ", id(ix), type(id(ix)))

L.append(7)
print(L, " : ", id(L), type(id(L)))
for ix in L:
    print(ix, " : ", id(ix), type(id(ix)))
print("\n")

L2 = L
print(L2, " : ", id(L2), type(id(L2)))
for ix in L2:
    print(ix, " : ", id(ix), type(id(ix)))

L2.pop(0)
print(L2, " : ", id(L2), type(id(L2)))
for ix in L2:
    print(ix, " : ", id(ix), type(id(ix)))
print(L, " : ", id(L), type(id(L)))
for ix in L:
    print(ix, " : ", id(ix), type(id(ix)))
print("\n")

L3 = L[:]
print(L3, " : ", id(L3), type(id(L3)))
for ix in L3:
    print(ix, " : ", id(ix), type(id(ix)))

L3[2] = 5
print(L3, " : ", id(L3), type(id(L3)))
for ix in L3:
    print(ix, " : ", id(ix), type(id(ix)))
print(L, " : ", id(L), type(id(L)))
for ix in L:
    print(ix, " : ", id(ix), type(id(ix)))
print("\n")


# tuple

t = (2, 3, "c")
print(t, " : ", id(t), type(id(t)))
for ix in t:
    print(ix, " : ", id(ix), type(id(ix)))
print("\n")

t2 = t
print(t2, " : ", id(t2), type(id(t2)))
for ix in t2:
    print(ix, " : ", id(ix), type(id(ix)))
print("\n")

t = 3
print(t, " : ", id(t), type(id(t)))
print(t2, " : ", id(t2), type(id(t2)))
print("\n")


# list inside tuple

t = ([2, 3, 4], 5, "cc", "tt", 7)
print(t, type(t), " : ", id(t))
for key, value in enumerate(t):
    print(value, type(value), " in index ", key, " : ", id(value))
print("\n")


ml = [2, 3, 4]
print(ml, type(ml), " : ", id(ml))
for key, value in enumerate(ml):
    print(value, type(value), " in index ", key, " : ", id(value))
t = (ml, 5, "cc", "tt", 7)
print(t, type(t), " : ", id(t))
for key, value in enumerate(t):
    print(value, type(value), " in index ", key, " : ", id(value))
print("\n")

ml.append(10)
print(ml, type(ml), " : ", id(ml))
for key, value in enumerate(ml):
    print(value, type(value), " in index ", key, " : ", id(value))
print(t, type(t), " : ", id(t))
for key, value in enumerate(t):
    print(value, type(value), " in index ", key, " : ", id(value))
print("\n")


# function

y = 9
print(y, type(y), " : ", id(y))

wurzel = math.sqrt(y)
print(wurzel, type(wurzel), " : ", id(wurzel))
print("math  ", type(math), " : ", id(math))
print("math.sqrt  ", type(math.sqrt), " : ", id(math.sqrt))
print("\n")

def addone(num):
    out = num + 1
    return out


x = addone(6)
print("x =  ", x, type(x), " : ", id(x))
print("addone  ",  type(addone), " : ", id(addone))

Ao = addone
y = Ao(9)
print("y =  ", y, type(y), " : ", id(y))
print("addone  ", type(addone), " : ", id(addone))
print("Ao  ", type(Ao), " : ", id(Ao))
print("\n")


def factorial_of(n: int):
    """
    Calculates n! (n factorial)
    as Exercise 2
    :param n: (int) number as input for the factorial function
    :return: n factorial
    """
    start_time = dt.datetime.now()
    out_factorial = 1
    print("start_time =  ", start_time, type(start_time), " : ", id(start_time))
    print("out_factorial =  ", out_factorial, type(out_factorial), " : ", id(out_factorial))
    if n > 0:
        for ix in range(1, n+1):
            out_factorial *= ix
            print("ix =  ", ix, type(ix), " : ", id(ix))
            print("out_factorial =  ", out_factorial, type(out_factorial), " : ", id(out_factorial))

    time_duration = dt.datetime.now() - start_time
    print("time_duration =  ", time_duration, type(time_duration), " : ", id(time_duration))
    print("run time: ", time_duration.total_seconds(), " seconds ")

    return out_factorial


test_out = factorial_of(4)
print("test_out =  ", test_out, type(test_out), " : ", id(test_out))
print("factorial_of  ", type(factorial_of), " : ", id(factorial_of))
print("\n")

# import Tag7_Loesungen as t7L
test_out = t7L.fakulaet(5)
print("test_out =  ", test_out, type(test_out), " : ", id(test_out))
print("fakulaet  ", type(t7L.fakulaet), " : ", id(t7L.fakulaet))
print("\n")

test_out = t7L.fakulaet(4)
print("test_out =  ", test_out, type(test_out), " : ", id(test_out))
print("fakulaet  ", type(t7L.fakulaet), " : ", id(t7L.fakulaet))
print("\n")


# lambda function
add_l_one = lambda x: x+1
test_out = add_l_one(7)
print("test_out =  ", test_out, type(test_out), " : ", id(test_out))
print("addone  ", type(addone), " : ", id(addone))
print("add_l_one  ", type(add_l_one), " : ", id(add_l_one))
print("\n")

cubic_of = lambda x: x**3
test_out = cubic_of(2)
print("test_out =  ", test_out, type(test_out), " : ", id(test_out))
print("cubic_of  ", type(cubic_of), " : ", id(cubic_of))
print("\n")

diff_of = lambda x,y: x-y
test_out = diff_of(7, 5)
print("test_out =  ", test_out, type(test_out), " : ", id(test_out))
print("diff_of  ", type(diff_of), " : ", id(diff_of))
print("\n")


# map function
mylist = [x for x in range(8, 20)]
results = map(lambda x: x**2, mylist)
print(results, type(results))
print(list(results))

ml2 = ["Hallo", "Bla"]
results = map(str.upper, ml2)
print(results, type(results))
print(list(results))

results = map(math.sqrt, mylist)
print(results, type(results))
print(list(results))




