# Python, using Anaconda environment
# Week 1, Day 5

# list
mylist = [1, 5, 7, 3, 9]
print(mylist)
print(mylist[1:3], type(mylist[1:3]))
mylist.remove(5)
print(mylist)
print("\n")

mylist = [1, 5, 7, 3, 9, 5]
print(mylist)
mylist.remove(5)
print(mylist)
print("\n")

mylist = [1, 5, "hallo", "blubb", 9, 5]
print(mylist)
mylist.remove("blubb")
print(mylist)
print("\n")

mylist = [1, 5, "hallo", "blubb", [9, 8], 5]
print(mylist)
mylist.remove([9, 8])
print(mylist)
print("\n")

mylist = [1, 2, 3, 4]
print(mylist)
mylist.append("5")
print(mylist)
mylist.insert(1, 99)
print(mylist)
mylist.extend([6, 7, 8])
print(mylist)
print("\n")

# mylist.append([7, 9, 3])
# mylist.extend([7, 9, 3])
print(mylist)
print("\n")

mylist = [1, "banana", 5]
print(mylist[1])
print("banana" in mylist)
print("\n")

mylist1 = [1, 2, 3, 4]
mylist2 = [2, 3, 5, 7]
print(mylist1 + mylist2)
print(mylist1*2)
print(mylist1*5)
print("\n")

# tuple
mytuple = (1, 2, 3, 4)
print(mytuple, type(mytuple))
mytuple = 1, 2, 3, 4
print(mytuple, type(mytuple))

print(mytuple[2], type(mytuple[2]))
print("\n")

# dict - dictionary
mydict = {"Teilnehmer": "Meyer", "Alter": 31}
print(mydict, type(mydict))
print(mydict["Teilnehmer"], type(mydict["Teilnehmer"]))
print(mydict["Alter"], type(mydict["Alter"]))
mydict["Haarfarbe"] = "Kupfer"
print(mydict)
print("\n")

mydict = {"Teilnehmer" : "Meyer", "Alter": 31, "Alter": 7}
print(mydict)
print("\n")

# Sortieren von list
mylist = [2, 1, 4, 3, 15, 98, 6, 31, -5, 0]
print(mylist)
print(sorted(mylist), type(sorted(mylist)))
print("\n")

# mylist = [2, 1, 4, 3, 15, [9, 7], 98, 6, 31, -5, 0]
# print(mylist)
# print(sorted(mylist), type(sorted(mylist)))  # TypeError: '<' not supported between instances of 'list' and 'int'
# print("\n")

# mylist = [2, 1, 4, 3, 15, "shollaw", 98, 6, 31, -5, 0]
# print(mylist)
# print(sorted(mylist), type(sorted(mylist)))  # TypeError: '<' not supported between instances of 'str' and 'int'
# print("\n")

mylist = [2, 1, 4, 3, 15, 98, 6, 31, -5, 0]
print(mylist)
print(sorted(mylist, reverse=True), type(sorted(mylist, reverse=True)))
print("\n")

mylist = ["a", "B", "c", "D"]
print(mylist)
print(sorted(mylist))
print(sorted(mylist, reverse=True))
print("\n")

mylist = ["Aa", "aa", "aA", "AA"]
print(mylist)
print(sorted(mylist))
print("\n")

mylist = ["aaaaa", "BB", "c", "DDDD"]
print(mylist)
print(sorted(mylist))
print(sorted(mylist, key=len))
print(sorted(mylist, key=str.upper))
print(sorted(mylist, key=str.lower))
print("\n")

mylist = [2, 1, 4, 3, 4]
print(mylist)
x = mylist.index(4)
print(x)
mylist.pop(0)
print(mylist)
print("\n")

mylist = [2, 1, 4, 3, 4]
print(mylist)
mylist.pop(mylist.index(3))
print(mylist)
print("\n")

mylist = [2, 1, 4, 3, 4]
print(mylist)
mylist.remove(4)
print(mylist)
print("\n")

mylist = [2, 1, 4, 3, 4]
print(mylist)
mylist.sort()
print(mylist)
mylist.clear()
print(mylist)
print("\n")

mylist = [2, 1, 4, 3, 4]
mylist1 = mylist.copy()
mylist.sort()
print(mylist)
print(mylist1)
print("\n")

print(*mylist)
print(*mylist, sep=";")
print(*mylist, sep="; ", end="\n")
print("\n")

# slicing
mylist = [2, 3, 4, 5]
print(mylist)
mylist1 = mylist[:2]
print(mylist1)
print("\n")

# Kopie
mylist2 = mylist  # kein Kopie
print(mylist)
mylist2.remove(3)
print(mylist)
print(mylist2)
print("\n")

mylist = [2, 3, 4, 5]
print(mylist)
mylist1 = mylist.copy()
mylist2 = mylist[:]
print(mylist)
print(mylist1)
print(mylist2)
print("\n")
mylist1.remove(3)
mylist2.append(7)
print(mylist)
print(mylist1)
print(mylist2)
print("\n")

# dictionary
print(mydict)
print(mydict.get("Alter"))
mydict1 = mydict.copy()
mydict1.pop("Alter")
print(mydict1)
print("\n")

# try dictionary
mydict = {}
print(mydict, type(mydict))
key_string = "City"
value_string = "55"
mydict[key_string] = value_string
print(mydict, type(mydict))
print("\n")

# read info about machine learning
# read this:  https://www.alexanderthamm.com/de/blog/top-10-machine-learning-frameworks/
# read this:  https://www.computerwoche.de/a/was-ein-machine-learning-engineer-koennen-muss,3615008
# read this:  https://www.it-jobuniverse.de/karriere-ratgeber/was-macht-ein-machine-learning-engineer-m-w-d-155.html
# read this:  https://techminds.de/jobprofile/machine-learning-engineer/
print("\n")
