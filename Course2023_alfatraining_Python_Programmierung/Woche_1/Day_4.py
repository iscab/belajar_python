# Python, using Anaconda environment
# Week  1, Day 4
import datetime as dt

# Exercise 1
print("# Exercise 1: how old is Shiva now?  ")

datum_jetzt = dt.datetime.now()
geburtstag_shiva = dt.datetime(2000,8,1)
print("Today:  ", datum_jetzt, type(datum_jetzt))
print("Shiva's birthday:  ", geburtstag_shiva, type(geburtstag_shiva))
print("\n")

datum_diff = datum_jetzt - geburtstag_shiva
print("difference in days:  ", datum_diff, type(datum_diff))

datum_diff_year = int(datum_diff.days/365.25)
print("Shiva's age:  ", datum_diff_year, type(datum_diff_year))
print("\n")

# Exercise 2
print("# Exercise 2: open txt file and read individual rows  ")

myTextFile = open("Datumsspass.txt", "r", encoding="utf-8")
print(myTextFile, type(myTextFile))
myText_list = myTextFile.readlines()
print(myText_list, type(myText_list))
myTextFile.close()
print("\n")

# Exercise 3
print("Exercise 3: standardize dates  \n")

# print("length:  ", len(myText_list))
newText = myText_list[0]
standard_date_format = "%A, %d %B %Y\n"

date_temp = dt.datetime.strptime(myText_list[1], "%d.%m.%y\n")
date_1_str = date_temp.strftime(standard_date_format)
newText += date_1_str

date_temp = dt.datetime.strptime(myText_list[2], "%A, the %d. %B %Y\n")
date_2_str = date_temp.strftime(standard_date_format)
newText += date_2_str

date_temp = dt.datetime.strptime(myText_list[3], "%d-%m-%Y\n")
date_3_str = date_temp.strftime(standard_date_format)
newText += date_3_str

date_temp = dt.datetime.strptime(myText_list[4], "%m/%d/%Y\n")
date_4_str = date_temp.strftime(standard_date_format)
newText += date_4_str

date_temp = dt.datetime.strptime(myText_list[5], "%d-%m/%y")
date_5_str = date_temp.strftime(standard_date_format)
newText += date_5_str

print(newText)
print("\n")

# Exercise 4:
print("Exercise 4: save the formatted dates to a new file  \n")

myTextFile = open("Datums_neue.txt", "w", encoding="utf-8")
myTextFile.write(newText)
myTextFile.close()
print("\n")

# Exercise 5:
print("Exercise 5:  ")
list_of_columns = ["name", "favourite_colour", "favourite_animal"]
name_str = input("What is your name?  ")
fav_colour_str = input("What is your favourite colour?  ")
fav_animal_str = input("What is your favourite animals?  ")

# csv: comma
str_for_csv = list_of_columns[0] + ","
str_for_csv += list_of_columns[1] + ","
str_for_csv += list_of_columns[2] + "\n"
str_for_csv += name_str + ","
str_for_csv += fav_colour_str + ","
str_for_csv += fav_animal_str

# csv2: semicolon
str_for_csv2 = list_of_columns[0] + ";"
str_for_csv2 += list_of_columns[1] + ";"
str_for_csv2 += list_of_columns[2] + "\n"
str_for_csv2 += name_str + ";"
str_for_csv2 += fav_colour_str + ";"
str_for_csv2 += fav_animal_str

# print(str_for_csv)
myTextFile = open("favourites.csv", "w", encoding="utf-8")
myTextFile.write(str_for_csv)
myTextFile.close()

myTextFile = open("favourites.txt", "w", encoding="utf-8")
myTextFile.write(str_for_csv)
myTextFile.close()

myTextFile = open("favourites2.csv", "w", encoding="utf-8")
myTextFile.write(str_for_csv2)
myTextFile.close()

myTextFile = open("favourites2.txt", "w", encoding="utf-8")
myTextFile.write(str_for_csv2)
myTextFile.close()
print("\n")
