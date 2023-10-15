# Python, using Anaconda environment
# Week 2, Day 7
import datetime as dt
import time
import math

def sum_till(x: int):
    """
    Calculate the sum of 0 to X
    as Exercise 1 oldies
    :param x: (int) the number at the end
    :return: the sum of 0 to x
    """
    start_time = dt.datetime.now()
    mySum = 0
    for ix in range(0, x+1):
        mySum += ix

    time_duration = dt.datetime.now() - start_time
    print("run time: ", time_duration.total_seconds() , " seconds ")

    return mySum


def area_of_square(edge_length):
    """
    Calculate the area of the square, given the edge length
    as Exercise 2 oldies
    :param edge_length: (float/int) the edge length of the square
    :return: the area of the square
    """
    return edge_length**2


def quadrat_label(txtstr: str, number):
    """
    Calculate the square of a number
    as Exercise 1
    :param txtstr: (str) input text as a key
    :param number: (int/float) a number
    :return:
    """
    print(f"The input text is {txtstr}")
    out_number = number**2
    out_pair = {txtstr: out_number}

    return out_number, out_pair


def factorial_of(n: int):
    """
    Calculates n! (n factorial)
    as Exercise 2
    :param n: (int) number as input for the factorial function
    :return: n factorial
    """
    start_time = dt.datetime.now()
    out_factorial = 1
    if n > 0:
        for ix in range(1, n+1):
            out_factorial *= ix

    time_duration = dt.datetime.now() - start_time
    print("run time: ", time_duration.total_seconds(), " seconds ")

    return out_factorial


def odd_or_even(*mylists):
    start_time = dt.datetime.now()
    # print(mylists)
    odd_list = []
    even_list = []
    for mylist in mylists:
        for ix in mylist:
            # print(ix, type(ix))
            if ix % 2 == 0:
                even_list.append(ix)  # even numbers
            else:
                odd_list.append(ix)  # odd numbers

    time_duration = dt.datetime.now() - start_time
    print("run time: ", time_duration.total_seconds(), " seconds ")

    return even_list, odd_list


def character_count(strText: str):
    """
    calculate capitals, small letters, and whitespaces in a string,
    as Exercise 6
    :param strText: (str) text as input
    :return: (dict) the result of counting
    """
    start_time = dt.datetime.now()
    out_dict = {}
    num_of_capital = len([x for x in strText if x.isupper()])
    num_of_small_letter = len([x for x in strText if x.islower()])
    num_of_space = strText.count(" ")

    out_dict["capital"] = num_of_capital
    out_dict["small letter"] = num_of_small_letter
    out_dict["whitespace"] = num_of_space

    time_duration = dt.datetime.now() - start_time
    print("run time: ", time_duration.total_seconds(), " seconds ")

    return out_dict


# read this: https://realpython.com/python-kwargs-and-args/


def cone_volume(radius: float = 1.0, height: float = 1.0):
    return math.pi * radius**2 * height / 3.0


def sphere_volume(radius: float = 1.0):
    return math.pi * radius**3 * 4.0 / 3.0


def cuboid_volume(length: float = 1.0, width: float = 1.0, height: float = 1.0):
    return length * width * height


def cylinder_volume(radius: float = 1.0, height: float = 1.0):
    return math.pi * radius**2 * height


def shape_volume(fkt, *args):
    volume = fkt(*args)
    return volume



