# Python, using Anaconda environment
# Week 1, Day 3
import datetime as dt
import time
import this as zen

# String slicing
x = "Hallo Welt!"

print(x)

# positive index
print(x[0:9])
print(x[0:10])
print(x[6:11])
print("\n")

# negative index
print(x[-4:-1])
print(x[-1:-4])  # nothing comes out
print(x[-10:-1])
print(x[-1:-10])  # nothing comes out
print("\n")

x = "Shiva ist die Beste!"

# positive index
print(x[0])
print(x[3])
print(x[5:8])
print(x[:8])
print(x[18:])
print("\n")

print(x[10:19])
print("\n")

# negative index
print(x[:-4])
print(x[-18:])
print(x[-1:])
print("\n")

x = "Shiva ist die Beste!"

# mixed index
print(x[-5:3])  # nothing comes out
print(x[-9:-3])
print(x[-9:-3:2])
print(x[::-1])
print("\n")

# read this:  https://stackoverflow.com/questions/48776238/python-slice-operator-negative-start-combined-with-positive-stop

x = "   Shiva ist die Beste!   "

# strings modification
print(x.lower())
print(x.upper())
print(x.strip())
print(x.replace("e", "3"))
print(x.split("ist"))
print("\n")

x = "Jojon".upper()
print(x)
print("\n")

# Strings Zusammenfügen
x = "Einhorn"
y = "Power"

z1 = x + y
print(z1)
# z2 = x y  # not working
# print(z2)
z3 = x + " " + y
print("\n")

# Strings and Numbers

# print("Ich bin: " + 420)  # TypeError: can only concatenate str (not "int") to str
print("Ich bin: ",  420)
print("\n")

b = "Mein Drachenbaum ist {} Jahre alt"
print(b.format(18))
print(b.format(18.5))

b = "Mein Drachenbaum ist {} Jahre alt und ich bin mindestens {} Jahre aelter"
print(b.format(18, 10))
# print(b.format(18))  # IndexError: Replacement index 1 out of range for positional args tuple

b = "Mein Drachenbaum ist {1} Jahre alt und ich bin mindestens {0} Jahre aelter"  # 0 and 1 are index numbers
print(b)
print(b.format(18, 10))
print(b.format(10, 18))
print("\n")

bsp ="a backlash \\"
print(bsp)
print("\n")

zeit_jetzt = dt.datetime.now()
print(zeit_jetzt, type(zeit_jetzt))
time.sleep(0.5)
print("some message")
other_zeit = dt.datetime.now()
print(other_zeit, type(other_zeit))
print(zeit_jetzt, type(zeit_jetzt))

print(zeit_jetzt.year, type(zeit_jetzt.year))
print(zeit_jetzt.month, type(zeit_jetzt.month))
print(zeit_jetzt.day, type(zeit_jetzt.day))
print("\n")

# Aufgabe 3: Python Zen
# import this as zen
print(zen.s, type(zen.s))
print(zen.d, type(zen.d))

print(zen, type(zen))

# read this:  https://stackoverflow.com/questions/23794344/how-can-i-return-pythons-import-this-as-a-string
zen_string = "".join([zen.d.get(c, c) for c in zen.s])
print(zen_string, type(zen_string))
print("\n")
