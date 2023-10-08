# Python, using Anaconda environment
# Week 1, Day 4
import datetime as dt
import io, contextlib

# try yesterday contextlib for Exercise 3
with contextlib.redirect_stdout(zen := io.StringIO()):
    import this
print(type(zen))
zen_string = zen.getvalue()
print(zen_string, type(zen_string))
print("\n")
# what is assignment ":=" ?
# read this:  https://stackoverflow.com/questions/26000198/what-does-colon-equal-in-python-mean
# read this:  https://docs.python.org/3/whatsnew/3.8.html
# read this:  https://www.w3schools.com/python/python_operators.asp

# datetime
yesterday = dt.datetime(2023, 5, 25)
print(yesterday, type(yesterday))
print("\n")

# date
today_exp = dt.date.today()
print(today_exp, type(today_exp))
print(today_exp.weekday(), type(today_exp.weekday()))
print("\n")

geburtstag_she = dt.date(2000, 8, 2)
print(geburtstag_she, type(geburtstag_she))
print("\n")

# time
Pause = dt.time(10, 5, 8)
print(Pause, type(Pause))
Pause = dt.time(10, 5, 8, 7437)
print(Pause, type(Pause))
print("\n")

# combine
new_date = dt.datetime.combine(geburtstag_she, Pause)
print(new_date, type(new_date))
print("\n")

# date as string
test_date = "05-01-2010 11:08:00"
new_date = dt.datetime.strptime(test_date, "%d-%m-%Y %H:%M:%S")
print(new_date, type(new_date))

date_string = new_date.strftime("%A, %d %B %Y")
print(date_string, type(date_string))
print("\n")

# Exercise
some_date = "15.07.1980 20:45:00"
dt_date = dt.datetime.strptime(some_date, "%d.%m.%Y %H:%M:%S")
print(dt_date, type(dt_date))

date_string = dt_date.strftime("%A, %d %B %Y")
print(date_string, type(date_string))
date_string = dt_date.strftime("%A, %d-%m-%Y")
print(date_string, type(date_string))
date_string = dt_date.strftime("%B %d, %Y")
print(date_string, type(date_string))
print("\n")

# file open
# mytext = open("bsp.txt","r")
mytext = open("bsp.txt","r", encoding="utf-8")
print(mytext, type(mytext))
print("\n")

# mytext_str = mytext.read()
mytext_str = mytext.read(3)
print(mytext_str, type(mytext_str))
print("\n")

mytext_lines = mytext.readlines()
print(mytext_lines, type(mytext_lines))
print(mytext_lines[0], type(mytext_lines[0]))
print(mytext_lines[2], type(mytext_lines[2]))
print("\n")
mytext.close()

# input
eingabe = input("Any comments?  ")
print(eingabe, type(eingabe))
print("\n")

# output
print("my file is here.  ", file=open("new_text.txt","w"))
print("\n")
