# Python, using Anaconda environment
# Week 3, Day 12
import helper_func.day12func as hfun

# Fehlertypen: types of error

x = 5
"""if x > 0:
    raise Exception("Number has to be below 0")"""

"""if not type(x) == str:
    raise TypeError("Input has to be string")"""


x = 4
xx = hfun.sum_till(x)
print(xx)
print("\n")

# class
# read this:  https://docs.python.org/3/tutorial/classes.html
# read this:  https://www.w3schools.com/python/python_classes.asp



dog1 = hfun.Dog("Shiva", "Mud", 7, "mostly black")

print(dog1.name)
print(dog1.breed)
print(dog1.age)
print(dog1.color)
print("\n")

dog0 = hfun.Dog()

print(dog0.name)
print(dog0.breed)
print(dog0.age)
print(dog0.color)
print("\n")

dog2 = hfun.Dog("Jojon", "Pugs", 3, "brown")

print(dog2.name)
print(dog2.breed)
print(dog2.age)
print(dog2.color)
print("\n")

print(dog1.color)
dog1.color = "Pink"
print(dog1.color)
print("\n")

print(dog1.age)
del dog1.age
# print(dog1.age)  # AttributeError: 'Dog' object has no attribute 'age'
print("\n")

print(dog2, type(dog2))
print(dog2.age)
del dog2
# print(dog2, type(dog2))  # NameError: name 'dog2' is not defined. Did you mean: 'dog1'?
# print(dog2.age)  # NameError: name 'dog2' is not defined. Did you mean: 'dog1'?
print("\n")


shinobi1 = hfun.Ninja("Yayan Ruhiyan", "karambit", "Silat", 4)
print(shinobi1.name)
print(shinobi1.weapon)
print(shinobi1.jutsu)
print(shinobi1.level)
print(shinobi1, type(shinobi1))
print("\n")

shinobi0 = hfun.Ninja()
print(shinobi0.name)
print(shinobi0.weapon)
print(shinobi0.jutsu)
print(shinobi0.level)
print(shinobi0, type(shinobi0))
print("\n")

