# Python, using Anaconda environment
# Week 3, Day 12
import helper_func.day12func as hfun
import random


# Exercise 1
print("Exercise 1:  ")
myTextFile = open("zahlen.txt", "r", encoding="utf-8")
myText = myTextFile.read()
myTextFile.close()

print(myText, type(myText))
myTextList = myText.split(",")
print(myTextList, type(myTextList))
myList = [int(x) for x in myTextList]
print(myList, type(myList))
print("\n")


# Exercise 2
print("Exercise 2 : math substitute")
x = [1, 3, 5, 7]
# x = ["hey", "mari", "kemari"]

x_mean = hfun.theMean(x)
x_sum = hfun.theSum(x)
x_prod = hfun.theProduct(x)

print(x_mean, type(x_mean))
print(x_sum, type(x_sum))
print(x_prod, type(x_prod))
print("\n")

# Exercise Dungeon Crawler

maze1 = hfun.Maze(5,8)
print(maze1.num_of_wormhole)