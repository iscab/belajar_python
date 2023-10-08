# Python, using Anaconda environment
# Week 3, Day 12

def sum_till(x):
    """
    the sum from 0 to X
    :param x: (int) number at the end
    :return: mySum as the sum
    """
    if not type(x) == int:
        raise TypeError(f"The input should be an integer, and {x} is not")
    if x > 1000:
        raise OverflowError(f"The number {x} is too large")

    mySum = 0
    for ix in range(0, x+1):
        mySum += ix

    return mySum


class Dog:
    def __init__(self, name="Blacky", breed="Mongrel", age=6, color="mostly black"):
        self.name = name.upper()
        self.breed = breed
        self.age = age + 1
        self.color = color

    def __str__(self):
        return f"{self.name}, {self.age}. A {self.breed} with the color {self.color}"


class Ninja:
    def __init__(self, name="Naruto", weapon="Shuriken", jutsu= "Kage Bunshin", level=5):
        self.name = name
        self.weapon = weapon
        self.jutsu = jutsu
        self.level = level
        self.about = f"{self.name}, ninja level {self.level}, expert of {self.weapon} and {self.jutsu}"

    def __str__(self):
        return self.about


def clean_list(input_list):
    if not isinstance(input_list, list):
        raise TypeError("Input has to be a list")
    else:
        for ix in input_list:
            if not isinstance(ix, int):
                raise TypeError("the list should contain integer")
            else:
                return input_list


def theSum(input_list):
    input_list = clean_list(input_list)
    mySum = 0
    for ix in input_list:
        mySum += ix

    return mySum


def theMean(input_list):
    input_list = clean_list(input_list)
    num_element = len(input_list)
    sum_element = theSum(input_list)

    return sum_element/num_element


def theProduct(input_list):
    input_list = clean_list(input_list)
    myProduct = 1
    for ix in input_list:
        myProduct *= ix

    return myProduct


class Walker:
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y


class Maze:
    def __init__(self, dim_x=8, dim_y=8):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_of_wormhole = min(self.dim_x, self.dim_y)


