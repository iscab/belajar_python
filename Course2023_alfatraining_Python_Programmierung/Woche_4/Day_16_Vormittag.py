# Python, using Anaconda environment
# Week 4, Day 16
import helper_func.day16func as hfun
from dataclasses import asdict, astuple
import tkinter as tk

bsp_teilnehmer = hfun.Teilnehmer("John", 23, "Iron Smith")
bsp_teilnehmer.my_age()
bsp_teilnehmer.my_job()
print(bsp_teilnehmer, type(bsp_teilnehmer))
print("\n")

bsp_teilnehmer1 = hfun.Member("John", 23, "Iron Smith")
bsp_teilnehmer1.my_age()
bsp_teilnehmer1.my_job()
print(bsp_teilnehmer1, type(bsp_teilnehmer1))
print("\n")


print(bsp_teilnehmer, type(bsp_teilnehmer))

test_tuple = astuple(bsp_teilnehmer)
print(test_tuple, type(test_tuple))

test_dict = asdict(bsp_teilnehmer)
print(test_dict, type(test_dict))
print("\n")


# generator & iterator

mylist = ["a", "b", "c", "d"]
print(mylist, type(mylist))
mylist = iter(mylist)
print(mylist, type(mylist))
print(next(mylist))
print(next(mylist))
print(next(mylist))
print(next(mylist))
# print(next(mylist))  # Error:   StopIteration
print("\n")


beispiel = hfun.durchzahlen("abc")
for ix in range(10):
    print(next(beispiel), end=" ")
print("\n")


def counter(n):
    for ix in range(n + 1):
        # yield ix
        # yield ix**2
        # yield ix**0.5
        yield (ix ** 2 + 1) ** -0.5


for num in counter(10):
    print(num, type(num))

# tkinter for Python GUI

# try this online:  https://visualtk.com/

mywindow = tk.Tk()
mywindow.geometry("600x400+100+50")
mywindow.title("Jendela Rumah Kita")

mywindow.mainloop()
