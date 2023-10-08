# Python, using Anaconda environment
# Week 2, Day 8
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
    mySum = 0
    for ix in range(0, x+1):
        mySum += ix

    return mySum



