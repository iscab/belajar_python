# Python, using Anaconda environment
# Week 3, Day 13
import helper_func.day13func as hfun


# methods in classes

dog1 = hfun.Dog("shiva", "Mud", 7, "mostly black")

print(dog1)
print(dog1.name)
print(dog1.breed)
print(dog1.eat("banana"))
print("\n")


shinobi1 = hfun.Ninja()

print(shinobi1)
print("before:  ", shinobi1.my_position())
shinobi1.initial_position(3,4)
print("after:  ", shinobi1.my_position())
shinobi1.go_left()
print("go left:  ", shinobi1.my_position())
shinobi1.go_right()
print("go right:  ", shinobi1.my_position())
shinobi1.go_down()
print("go down:  ", shinobi1.my_position())
shinobi1.go_up()
print("go up:  ", shinobi1.my_position())
print("\n")

dog0 = hfun.Dog()

dog_z = dog1 + dog0
print(dog_z)
dog_z = dog0 + dog1
print(dog_z)
print("\n")


# mutable and immutable objects
def print_list(input_list):
    for ix in input_list:
        print(ix, " :  ", id(ix))


mylist = [2, 6, 2, 4, 9]
print(mylist, " :  ", id(mylist))
print_list(mylist)
mylist[2] = "test"
print(mylist, " :  ",  id(mylist))
print_list(mylist)
mylist.append(42)
print(mylist, " :  ",  id(mylist))
print_list(mylist)
print("\n")
