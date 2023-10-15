# Python, using Anaconda environment
# Week 3, Day 11
from functools import reduce

# Exercise 1
print("Exercise 1: read file")
MyTextFile = open("woerter.txt", "r", encoding="utf-8")
MyTextList = MyTextFile.readlines()
MyTextFile.close()

print(MyTextList, type(MyTextList))
print("\n")

# Exercise 2
print("Exercise 2: cleaning the string list")
SaveText = ""
MyNewTextList = []
for idx, myText in enumerate(MyTextList):
    # print(myText)
    myText = myText.strip()
    myText = myText.replace("\t", "")
    # myText = myText.capitalize()
    myText = myText.replace(" ", "")
    # print(myText)
    myText = myText.split(",")
    # print(myText)
    myNewText = ""
    for myWord in myText:
        myWord = myWord. capitalize()
        myNewText += myWord + ","
        MyNewTextList.append(myWord)
    MyTextList[idx] = myNewText
    SaveText += myNewText
print(MyTextList, type(MyTextList))
print(SaveText, type(SaveText))
print(MyNewTextList, type(MyNewTextList))

# saving
MyTextFile = open("woerter_neu.txt", "w", encoding="utf-8")
MyTextFile.write(SaveText)
MyTextFile.close()
print("\n")

# Exercise 3
print("Exercise 3: ")
dict_of_word_length = {x: len(x) for x in MyNewTextList}
print(dict_of_word_length)

# read this:  https://www.w3schools.com/python/python_lists_comprehension.asp

"""for myText in MyNewTextList:
    print(myText)
    print(myText.find("n"))"""
n_word_list = [x for x in MyNewTextList if x.find("n") > 0]
print(n_word_list)
fe_word_list = [x for x in MyNewTextList if x.find("fe") > 0]
print(fe_word_list)
dr_word_list = [x for x in MyNewTextList if x.find("dr") > 0]
print(dr_word_list)
new_word_list = [x for x in MyNewTextList if x.find("dr") > 0 or x.find("fe") > 0 or x.find("n") > 0]
print(new_word_list)
print("\n")

# Exercise 4
print("Exercise 4: ")
MyTextFile = open("buchstaben.txt", "r", encoding="utf-8")
MyBuchstaben = MyTextFile.read()
MyTextFile.close()

print(MyBuchstaben, type(MyBuchstaben))
MyBuchstaben = MyBuchstaben.replace(",", "")
MyBuchstaben = MyBuchstaben.replace(".", "")
MyBuchstaben = MyBuchstaben.replace("!", "")
print(MyBuchstaben, type(MyBuchstaben))
MyTextList = MyBuchstaben.split(" ")
print(MyTextList, type(MyTextList))
# from functools import reduce
# MyText = reduce(lambda x, y: x + y, MyBuchstaben)
# print(MyText, type(MyText))
print("\n")

# Exercise 5
print("Exercise 5: ")
MyTextFile = open("woerter_neu.txt", "r", encoding="utf-8")
MyText = MyTextFile.read()
MyTextFile.close()
print(MyText, type(MyText))

# this is wrong
# TODO: fix it or forget it
input_text = "lu"
xxx = input_text.lower().count("lu")
print(xxx)

print("\n")



