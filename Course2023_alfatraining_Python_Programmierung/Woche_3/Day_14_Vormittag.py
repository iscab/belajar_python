# Python, using Anaconda environment
# Week 3, Day 14
import helper_func.day14func as hfun

# immutable objects

mytuple = (34, 32, 67)
print(mytuple, " : ", id(mytuple))
for ix in mytuple:
    print(ix, " : ", id(ix))
print("\n")

mytuple = (34, 25, 67)
print(mytuple, " : ", id(mytuple))
for ix in mytuple:
    print(ix, " : ", id(ix))
print("\n")


x = 13
mytuple1 = (1, x, 45, 89)
print(x, " : ", id(x))
print(mytuple1, " : ", id(mytuple1))
for ix in mytuple1:
    print(ix, " : ", id(ix))
print("\n")

x = 2
print(x, " : ", id(x))
print(mytuple1, " : ", id(mytuple1))
for ix in mytuple1:
    print(ix, " : ", id(ix))
print("\n")

x = 3
mytuple1 = (1, x, 45, 89)
print(x, " : ", id(x))
print(mytuple1, " : ", id(mytuple1))
for ix in mytuple1:
    print(ix, " : ", id(ix))
print("\n")


x = ["a", "b"]
print(x, " : ", id(x))
mytuple2 = (3, x, 7, 2)
print(mytuple2, " : ", id(mytuple2))
for ix in mytuple2:
    print(ix, id(ix))
print("\n")

x.append("c")
print(x, " : ", id(x))
print(mytuple2, " : ", id(mytuple2))
for ix in mytuple2:
    print(ix, id(ix))
print("\n")

x = ["a", "b", "c"]
print(x, " : ", id(x))
print(mytuple2, " : ", id(mytuple2))
for ix in mytuple2:
    print(ix, id(ix))
print("\n")

x = ["a", "b", "c", "d"]
print(x, " : ", id(x))
print(mytuple2, " : ", id(mytuple2))
for ix in mytuple2:
    print(ix, id(ix))
print("\n")

x = mytuple2[1]
print(x, " : ", id(x))

x.append("f")
print(x, " : ", id(x))
print(mytuple2, " : ", id(mytuple2))
for ix in mytuple2:
    print(ix, id(ix))
print("\n")


# yy= mytuple.copy()  # AttributeError: 'tuple' object has no attribute 'copy'
# print(yy, " : ", id(yy))

# immutable Object doesn't need .copy
# mutable Object need .copy to define new Object

# Question: How about singleton?
# read this:  https://www.geeksforgeeks.org/singleton-pattern-in-python-a-complete-guide/
# read this:  https://python-patterns.guide/gang-of-four/singleton/
# read this:  https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons
# read this:  https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python


walker1 = hfun.Adventurer()
print(walker1, id(walker1))
print(walker1.name, id(walker1.name))
print(walker1.pos_x, id(walker1.pos_x))
print(walker1.pos_y, id(walker1.pos_y))
print("\n")

walker1.change_location(4,3)
print(walker1, id(walker1))
print(walker1.name, id(walker1.name))
print(walker1.pos_x, id(walker1.pos_x))
print(walker1.pos_y, id(walker1.pos_y))
print("\n")


# inheritance / Vererbung

myVehicle1 = hfun.MeansOfTransportation(20, 500, 3,"Transparent")
print(myVehicle1)
print(myVehicle1.speed)
print(myVehicle1.capacity)
print(myVehicle1.num_wheels)
print(myVehicle1.color)
myVehicle1.look()
print("\n")

myBike1 = hfun.MotorCycle(30, 200, 2,"Red")
print(myBike1)
print(myBike1.speed)
print(myBike1.capacity)
print(myBike1.num_wheels)
print(myBike1.color)
myBike1.look()
print("\n")

myCar1 = hfun.Car(120, 1600, 4, "White")
print(myCar1)
print(myCar1.speed)
print(myCar1.capacity)
print(myCar1.num_wheels)
print(myCar1.color)
myCar1.look()
print("\n")

myBike2 = hfun.Bicycle(35, 1, 2, "Black")
print(myBike2)
print(myBike2.speed)
print(myBike2.capacity)
print(myBike2.num_wheels)
print(myBike2.color)
myBike2.look()
print("\n")



