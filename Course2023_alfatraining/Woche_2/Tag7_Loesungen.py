# Musterloesungen Tag 7
import math
import random
import typing

print('============= Aufgabe 1 =============')


def mixfct(wort, zahl):
    if type(wort) == str:
        e1 = wort
    else:
        e1 = str(wort)
    if type(zahl) == int or type(zahl) == float:
        e2 = zahl
    else:
        print('Second Element not a number!')
    return {wort: zahl}


print('============= Aufgabe 2 =============')


def fakulaet(zahl):
    if type(zahl) == int or type(zahl) == float:
        out = 1
        for i in range(1, zahl + 1):
            out = out * i
    else:
        print(zahl, ' is not a number!')
        out = None # Damit out immer gegen ist!
    return out


print('============= Aufgabe 3 =============')


# Tag 6 Aufgabe 1:
def daysixexone(begin, end):
    out = []
    for i in range(begin, end + 1):
        if i % 2 == 0:
            out.append(i)
    return out


# Tag 6 Aufgabe 2:
def daysixextwo(i, t):
    while i <= 100:
        t += 1
        if i == 1:
            k = i
        else:
            k = k + i
        i += 1
        if k > 1000:
            k = k - i
            i -= 1
            break
        # Control statement:
        print(t)
        if t > 10000:
            RuntimeWarning(print('Infinity loop, Wuhu!'))
            break
    return {'Sum:': k, 'Last added number: ': i}


# Tag 6 Aufgabe 4:
def daysixexfour(beginn, end):
    prim_num = []
    prim = False
    for num in range(beginn, end):
        if num > 1:
            for i in range(2, num):
                if (num % i) == 0:
                    prim = True
            if prim:
                prim = False
            else:
                prim_num.append(num)
    return prim_num


# Tag 6 Aufgabe 5:
def daysixpw(password):
    if input(('Give me your password!')) == 'myPassword':
        print('The password is correct!')
    else:
        print('The password is incorrect')


print('============= Aufgabe 4 =============')


def checkbool(statement):
    out = False
    if type(statement) == str:
        info = ''
        for key, i in enumerate(statement):
            if i == ' ':
                continue
            if not i.isdigit():
                info = info + i
                if not statement[key + 1].isdigit():
                    info = info + '='
            if not info == '':
                numberleft = int(statement[:key])
                break
        length_operator = len(info)
        numberright = int(statement[key + length_operator:])
        # Brute force! Alles wird überprüft
        if info == '==':
            if numberleft == numberright:
                out = True
            else:
                out = False
        if info == '>=':
            if numberleft >= numberright:
                out = True
            else:
                out = False
        if info == '<=':
            if numberleft <= numberright:
                out = True
            else:
                out = False
        if info == '>':
            if numberleft > numberright:
                out = True
            else:
                out = False
        if info == '<':
            if numberleft < numberright:
                out = True
            else:
                out = False

    else:
        print(statement, ' is not a string!')
    return out


print(checkbool('4>=3'))


print('============= Aufgabe 5 =============')


def checkEvenOdd(*alist):
    even = []
    odd = []
    for sublist in alist:
        for i in sublist:
            if i % 2 == 0:
                even.append(i)
            else:
                odd.append(i)
    return [even, odd]


mylist = random.sample(range(100), 10)
print(mylist)
checkEvenOdd(mylist)


print('============= Aufgabe 6 =============')


def checkUpLowSpace(mystring: str) -> dict:
    upper = 0
    lower = 0
    space = 0
    for i in mystring:
        if i.isupper():
            upper += 1
        elif i.islower():
            lower += 1
        elif i == ' ':
            space += 1
    return {'Upper:': upper, 'Lower:': lower, 'Whitespace:': space}


print('============= Aufgabe 7 =============')


def sortingfun(athing, fkt=None, up=False):
    out = []
    if type(athing) == str or type(athing) == list:
        out = sorted(athing, key=fkt, reverse=up)
    else:
        print('Wrong type! Give me a string or list')
    return out


print('============= Aufgabe 8 =============')


def vol_cone(radius=0, heigth=0, diameter=0):
    if diameter > 0:
        radius = diameter / 2
    vol = 1 / 3 * math.pi * radius ** 2 * heigth
    return vol


def vol_sphere(radius=0, diameter=0):
    if diameter > 0:
        radius = diameter / 2
    vol = 4 / 3 * math.pi * radius ** 3
    return vol


def vol_cuboid(a=0, b=0, c=0):
    vol = a * b * c
    return vol


def vol_cylinder(radius=0, height=0, diameter=0):
    if diameter > 0:
        radius = diameter / 2
    vol = math.pi * radius ** 2 * height
    return vol


def volcalulation(text, radius=0, height=0, a=0, b=0, c=0, diameter=0):
    if diameter > 0:
        radius = diameter / 2
    out = 0
    if text.lower == 'cone':
        out = vol_cone(radius=radius)
    elif text.lower == 'sphere':
        out = vol_sphere(radius=radius)
    elif text.lower == 'cuboid':
        out = vol_cuboid(a=a, b=b, c=c)
    elif text.lower == 'cylinder':
        out = vol_cylinder(radius=radius, height=height)
    else:
        print('Nope, wrong shape. Try again')
    return {text: out}


print('============= Aufgabe 9 =============')


def listchecking(*args, position=0):
    if position != 0:
        print('The list at postion ', position, ' is: ', args[position])
        print('And now the elements:')
        for i in args[position]:
            print(i)
    else:
        for key, thelist in enumerate(args):
            print('The list at postion ', key, ' is: ', thelist)
            print('And now the elements:')
            for i in thelist:
                print(i)
