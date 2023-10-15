# Python, using Anaconda environment
# Week 2, Day 6


# looping
mystring = "Hello World!"
for ix in range(0, len(mystring)):
    print(ix)
print("\n")

myName = "Sapto Condro"
for ix in myName:
    print(ix)
print("\n")

mydict = {"name": "Sapto Condro", "city": "Wolfsburg", "country": "Germany"}
for ix in mydict:
    print(ix)
print("\n")

for ix, iy in mydict.items():
    print(ix, ":", iy)

print(mydict.items(), type(mydict.items()))
print("\n")

mylist = ["test", [1, 2, 3], 5, 6, 7, 8]
for ix in mylist:
    print(ix)
print("\n")

mylist2 = [1, 2, 3, 4]
for ix, iy in zip(mylist, mylist2):
    print("Mylist: ", ix, ";  Mylist2: ", iy)
print("\n")

for key, value in enumerate(mylist):
    print(key, ":", value)
print("\n")

for key, value in enumerate(mylist2):
    print(key, ":", value)
print("\n")

for key, value in enumerate(myName):
    print(key, ":", value)
print("\n")

mylist = [[0, 1], [7, 3], [9, 2]]
for ix, iy in mylist:
    print(ix, iy)
print("\n")

# if statement
myNumber = 5
if myNumber > 5:
    print(f"{myNumber} ist größer als 5 ")
elif myNumber == 5:
    print(f"{myNumber} ist genau gleich 5 ")
else:
    print(f"{myNumber} ist nicht größer als 5 ")
print("\n")

mylist =[]
myInputList = [1, 3, 5,  2, 8, 7]
for myNumber in myInputList:
    if myNumber > 5:
        strMessage = f"{myNumber} ist größer als 5 "
    elif myNumber == 5:
        strMessage = f"{myNumber} ist genau gleich 5 "
    else:
        strMessage = f"{myNumber} ist nicht größer als 5 "
    mylist.append(strMessage)
    print(strMessage)
print(mylist)
print("\n")

# list comprehension
mylist = [x for x in range(0,8)]
print(mylist)
mylist = [2*x for x in range(0,8)]
print(mylist)
mylist2 = [x for x in range(0,8) if x % 2 == 0 and x < 5]
print(mylist2)
print("\n")

mylist = [x**2 for x in range(0, 51) if x**2 <= 50]
print(mylist)
mylist = [x**2 for x in range(0, 51) if x**2 <= 50 and x**2 % 4 == 0]
print(mylist)
mylist = [x**2 for x in range(0, 51) if x**2 % 4 == 0]
print(mylist)
mylist = [x**0.5 for x in range(0, 51) if x**2 % 4 == 0]
print(mylist)
print("\n")

# while loop
ii = -5
while ii < 20:
    print(f"{ii} is less than 20")
    ii += 1
print("\n")

# break and continue
for ix in ["a", "b", "c", "d", "e"]:
    if ix == "c":
        break
    print(ix)
print("\n")

for ix in ["a", "b", "c", "d", "e"]:
    if ix == "c":
        continue
    print(ix)
print("\n")


# function

def beispiel():
    """
    Just prints some text
    :return: none
    """
    print("Some random text: Hello World!")

print(beispiel.__doc__)
beispiel()
print("\n")

def greet(Name):
    """
    Just print some text
    :param
    Name (str): just some name
    :return: None
    """
    print("Hello " + Name)


greet("Jojon")
print("\n")

def love(Name):
    """
    print text "I love you"
    :param
    Name (str): name
    :return: None
    """
    if isinstance(Name, str):
        print("I love you, " + Name)


love("Jojon")
love(5)
love("Heli")
print("\n")

def someMath(x, y, z):
    """
    summing x, y, and z and get the result as out

    :param x: (int/float)
    :param y: (int/float)
    :param z: (int/float)
    :return: out (int/float)
    """
    out = x + y + z
    return out


test = someMath(2, 3, 5)
print(test)

def someMoreMath(x, y, z):
    """
    addition and substraction
    :param x: (int/float)
    :param y: (int/float)
    :param z: (int/float)
    :return:
    out_1, out_2 (int/float)
    """
    out_1 = x + y + z
    out_2 = x - y - z
    return out_1, out_2


theAdd, theSub = someMoreMath(2, 3, 5)
print(theAdd, theSub)
testis = someMoreMath(2, 3, 5)
print(testis, type(testis))
print("\n")
