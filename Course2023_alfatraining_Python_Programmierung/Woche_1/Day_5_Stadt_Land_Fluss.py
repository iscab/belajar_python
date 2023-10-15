# Python, using Anaconda environment
# Week 1, Day 5
import datetime as dt
import math
import os

# Exercise Stadt Land Fluss
print("Welcome to the game Stadt Land Fluss!  \n")

# initial dictionaries and list
question_dict = {}
txt_file_dict = {"Name": "", "Letter": ""}
input_list = ["City", "Country", "River", "Fruit", "Vegetable", "Colour", "Drink", "Animal"]
# input_list = input_list[:3]  # debug purpose

# Name
question_str = "What is your name?  "
answer_str = input(question_str)
txt_file_dict["Name"] = answer_str
question_dict[question_str] = answer_str

# Letter
question_str = "Chose a letter to play:  "
answer_str = input(question_str)
txt_file_dict["Letter"] = answer_str
question_dict[question_str] = answer_str

# start to play the game
print("Let the game begin  \n")
letter_str = txt_file_dict["Letter"].upper()
start_time = dt.datetime.now()

for question_keyword in input_list:
    # print(question_keyword)
    question_str = f"Please input some {question_keyword.lower()} with \'{letter_str}\': "
    # print(question_str)
    answer_str = input(question_str)
    txt_file_dict[question_keyword] = answer_str
    question_dict[question_str] = answer_str
finish_time = dt.datetime.now()
print("\n")

# duration calculation
time_duration = finish_time - start_time
# print(time_duration, type(time_duration))
seconds_duration = math.ceil(time_duration.total_seconds())
txt_file_dict["duration_in_seconds"] = seconds_duration
question_dict["Your time in second is "] = seconds_duration

# is duration more than 60 s?
is_faster_than_a_minute = seconds_duration < 60
txt_file_dict["is_faster_than_a_minute"] = is_faster_than_a_minute
question_dict["Are you faster than 60 s?  "] = is_faster_than_a_minute

# Zusammenfassung
# print(question_dict)
# print(txt_file_dict)
out_string = "Stadt Land Fluss  \n"
column_names = ""
column_values = ""

for question_keyword, answer_str in txt_file_dict.items():
    # print(question_keyword, answer_str)
    column_names += question_keyword + ";"
    column_values += str(answer_str) + ";"
# change the end of the line
column_names = column_names[:-1] + "\n"
column_values = column_values[:-1]
# print(column_names)
# print(column_values)

for question_str, answer_str in question_dict.items():
    # print(question_str, answer_str)
    out_string += question_str + str(answer_str) + "  \n"
# print(out_string)

# prepare a folder/subdirectory
folder_name = os.getcwd()
folder_name = os.path.join(folder_name, "results")
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

# prepare file name
file_name = "stadt_land_flush"
file_name += start_time.strftime("_%H_%M")
csv_filename = file_name + ".csv"
csv_filename = os.path.join(folder_name, csv_filename)
# print(csv_filename)
txt_filename = file_name + ".txt"
txt_filename = os.path.join(folder_name, txt_filename)
# print(txt_filename)

# saving in the file
open(csv_filename, "w").write(column_names + column_values)
open(txt_filename, "w").write(out_string)
