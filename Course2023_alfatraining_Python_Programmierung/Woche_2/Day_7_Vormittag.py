# Python, using Anaconda environment
# Week 2, Day 7
import math

import funky
import helper_func.funkyxo as fck

print("Hello World! \n")

z = 10
print(z)

def myfunc(x, y):
    """
    Test function
    : global z: sum of x and y
    :param x: int
    :param y: int
    :return: multiply
    """
    global z
    z = x + y
    return x*y


test_z = myfunc(5, 3)
print(z)
print(test_z)
myfunc(3, 3)
print(z)
print("\n")

def myfkt(x, y):
    """
    quadratic equation
    :param x: (int)
    :param y: (int)
    :return: multiplication
    """
    global a
    a = x + y
    z = a * x
    return z


test_z = myfkt(7, 8)
print(test_z)
print(a)
print("\n")

# import funky
test_funky_sum = funky.mysum(2, 3, 5)
print(test_funky_sum)
print("\n")

# import helper_func.funkyxo as fck
txt_test = fck.nice("Pineapple")
print(txt_test, type(txt_test))
print("\n")


def beispiel(name, age):
    print(name, age)


beispiel(name="Shiva", age=7)
beispiel(age=7, name="Shiva")

beispiel("Shiva", 7)
beispiel(7, "Shiva")
print("\n")


def add_one(in_vector):
    new_list = []
    for value in in_vector:
        value += 1
        new_list.append(value)

    if isinstance(in_vector, tuple):
        out_vector = tuple(new_list)
    else:
        out_vector = new_list
    return out_vector


mylist = [1, 4, 5, 3]
my_new_list = add_one(mylist)
print(mylist)
print(my_new_list)

mytuple = (3, 6, 7, 2, 8)
my_new_tuple = add_one(mytuple)
print(mytuple)
print(my_new_tuple)
print("\n")


def add_one_again(list_vector):
    for key, value in enumerate(list_vector):
        list_vector[key] = value + 1


add_one_again(mylist)
print(mylist)
print("\n")


mylist = [x for x in range(8, 20)]

def beispielX(alist):
    alist.append("new")
    return alist


# mylist2 = beispielX(mylist)
mylist2 = beispielX(mylist[:])
print(mylist)
print(mylist2)
print("\n")


def beautiful_output(myText="Juhu!", txtWidth=40):
    """
    write text in a frame
    :param myText: (str) the input text
    :param txtWidth: (int) the width of the desired output
    :return:
    """
    min_indent = 4
    # width calculation
    min_width = len(myText) + 2 * min_indent
    if txtWidth > min_width:
        width = txtWidth
    else:
        width = min_width
    my_indent = int(0.5*(width + 1 - len(myText)))

    # write text
    out_string = ""
    frame = ""
    content = ""
    for ix in range(0,my_indent):
        frame += "-"
        content += " "

    content += myText
    for ix in range(0,len(myText)):
        frame += "-"

    for ix in range(0,my_indent):
        frame += "-"
        content += " "

    out_string += "+" + frame + "+\n"
    out_string += "+" + content + "+\n"
    out_string += "+" + frame + "+\n"

    return out_string


print(beautiful_output())
print(beautiful_output("This is the end"))
print(beautiful_output(txtWidth=20))
print(beautiful_output("Hello World!", 30))
print("\n")


def bsp(x):
    print(x)
    for ix in x:
        print(ix)


def bsp2(*x):
    print(x)
    for ix in x:
        print(ix)


mylist = [y for y in range(5)]
mylist2 = [y for y in range(6, 10)]

bsp(mylist)
bsp(mylist2)
bsp2(mylist, mylist2)
print("\n")

#  typing

def myfuntest(x: int, text: str) -> list:
    out = [x, text]
    return out


print(myfuntest(3, "haha"))
print(myfuntest(3.0, "haha"))
print(myfuntest(3,5))
print("\n")


def myfuntest2(x: int = 7, text: str = "Hello World!") -> list:
    out = [x, text]
    return out


print(myfuntest2())
print(myfuntest2(5))
print(myfuntest2(text="Jojon"))
print(myfuntest2(3,"Hi"))
print("\n")

# function as parameter
myfun = math.sqrt
print(myfun(9))
print("\n")

# functions as parameter

def loud(x):
    return x.upper()


def quiet(x):
    return x.lower()


def greet(fkt, text):
    out = fkt(text)
    print(out)


greet(loud, "Juhu")
greet(quiet, "Juhu")
print("\n")


def kuadrat(x: float):
    return x**2


def akar_kuadrat(x: float):
    return x**0.5


def number_manipulation(fkt, x):
    out = fkt(x)
    return out


print(number_manipulation(kuadrat, 2))
print(number_manipulation(akar_kuadrat, 2))
print(number_manipulation(kuadrat, 0.707))
print(number_manipulation(akar_kuadrat, 0.5))
print("\n")

# read this: https://realpython.com/python-kwargs-and-args/

