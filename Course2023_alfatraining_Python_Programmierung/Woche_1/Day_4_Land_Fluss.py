# Python, using Anaconda environment
# Week 1, Day 4
import datetime as dt
import math
import os

# Exercise Stadt Land Fluss
print("Welcome to the game Stadt Land Fluss!  \n")

letter_str = input("Please input a letter:  ")
letter_str = letter_str.upper()
name_str = input(f"Please input a name with {letter_str}:  ")
column_names = "Name;Letter;"
column_values = name_str + ";" + letter_str + ";"
letter_str = "\'" + letter_str + "\'"

start_time = dt.datetime.now()
city_str = input(f"Please input a city name with {letter_str}:  ")
column_names += "City;"
column_values += city_str + ";"

country_str = input(f"Please input a country name with {letter_str}:  ")
column_names += "Country;"
column_values += country_str + ";"

river_str = input(f"Please input a river name with {letter_str}:  ")
column_names += "River;"
column_values += river_str + ";"

fruit_str = input(f"Please input a fruit name with {letter_str}:  ")
column_names += "Fruit;"
column_values += fruit_str + ";"

vegetable_str = input(f"Please input a vegetable name with {letter_str}:  ")
column_names += "Vegetable;"
column_values += vegetable_str + ";"

colour_str = input(f"Please input a colour name with {letter_str}:  ")
column_names += "Colour;"
column_values += colour_str + ";"

drink_str = input(f"Please input a drink name with {letter_str}:  ")
column_names += "Drink;"
column_values += drink_str + ";"

finish_time = dt.datetime.now()
print("\n")

# time formats
standard_datetime_format = "at %H:%M:%S on %d %B %Y"

print("start time:  ", start_time, type(start_time))
start_time_text = start_time.strftime(standard_datetime_format)
print("start ", start_time_text, type(start_time_text))
column_names += "start_time;"
column_values += start_time_text + ";"
print("\n")

print("finish time:  ", finish_time, type(finish_time))
finish_time_text = finish_time.strftime(standard_datetime_format)
print("finish ", finish_time_text, type(finish_time_text))
column_names += "finish_time;"
column_values += finish_time_text + ";"
print("\n")

# duration calculation
time_duration = finish_time - start_time
print(time_duration, type(time_duration))
seconds_duration = math.ceil(time_duration.total_seconds())
# print(seconds_duration, type(seconds_duration))
# print(time_duration.seconds, type(time_duration.seconds))
# print(time_duration.total_seconds())
column_names += "duration_in_seconds;"
column_values += str(seconds_duration) + ";"
print("\n")

# is duration more than 60 s?
is_faster_than_a_minute = seconds_duration < 60
print(is_faster_than_a_minute, type(is_faster_than_a_minute))
column_names += "is_faster_than_a_minute\n"
column_values += str(is_faster_than_a_minute)

print("Columns:  ", column_names)
print("Values:  ", column_values)
print("\n")

# saving in a file in a subdirectory
folder_name = os.getcwd()
folder_name = os.path.join(folder_name, "results")
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

file_name = "stadt_land_flush.csv"
txt_file = os.path.join(folder_name, file_name)
print("file name:  ", txt_file, type(txt_file))
myTextFile = open(txt_file, "w", encoding="utf-8")
myTextFile.write(column_names + column_values)
myTextFile.close()
print("\n")
