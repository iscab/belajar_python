""" Covid 19 data processing

Procedure of data processing:
1. select variables of interest, from the data
2. clean the data
3. calculate descriptive statistics
4. visualize the data with diagrams or other images

data sources:  https://www.kaggle.com/datasets/hendratno/covid19-indonesia/data

environment: Python 3.11 under Anaconda 23.7.4

file version: 20:20 04.10.2023

"""

__author__ = "Ignatius S. Condro Atmawan"
__contact__ = "saptocondro@gmail.com"
__copyright__ = "alfatraining"
__date__ = "2023/10/05"

import os
import pandas as pd
import matplotlib.pyplot as plt

import helper_func.covid_data_process as cvd

# create object: Covid Data
myCovidData_clean = cvd.CovidData()
myCovidData_chaos = cvd.CovidData()

# prepare file name
FolderName = os.getcwd()
# print(FolderName)
FileName = "covid_19_indonesia_time_series_all.csv"
FileName1 = "Ignatius_covid_19_indonesia_time_series_all_chaos.csv"
FileName = os.path.join(FolderName, FileName)
FileName1 = os.path.join(FolderName, FileName1)
# print(FileName)
output_file_name = "covid19_clean_data.csv"
output_file_name = os.path.join(FolderName, output_file_name)

# read the CSV file
myCovidData_clean = myCovidData_clean.read_csv_data(FileName)
# print(myCovidData_clean.df)
print("\n")
myCovidData_chaos = myCovidData_chaos.read_csv_data(FileName1)
# print(myCovidData_chaos.df)
print("\n")

# columns
# print(myCovidData.df.columns, type(myCovidData.df.columns))
"""for col in myCovidData.df.columns:
    print(col, type(col))"""

print(myCovidData_clean.columns, type(myCovidData_clean.columns))
print(myCovidData_chaos.columns, type(myCovidData_chaos.columns))
# print(myCovidData_clean == myCovidData_chaos)

# myCovidData.df.describe()  # won't work because the data is not clean yet

mySelectedColumns = ["Date", "Location ISO Code", "Location", "New Cases", "New Deaths", "New Recovered", "New Active Cases", "Longitude", "Latitude"]
"""df_trial1 = myCovidData_chaos.df.filter(items=mySelectedColumns)
print(df_trial1)
print(df_trial1.columns)"""

myCovidData_chaos = myCovidData_chaos.select_column(mySelectedColumns)
print(myCovidData_chaos.df)
print(myCovidData_chaos.df.columns)
print(myCovidData_chaos.columns)
print("\n")

myCovidData_chaos = myCovidData_chaos.omit_empty_data()
print(myCovidData_chaos.df)
print(myCovidData_chaos.df.columns)
print(myCovidData_chaos.columns)
print("\n")
# myCovidData_chaos.save_csv_data(output_file_name)

myCovidData_clean = myCovidData_clean.select_column(mySelectedColumns)
myCovidData_clean = myCovidData_clean.omit_empty_data()
print(myCovidData_clean.df)
print(myCovidData_clean.df.columns)
print(myCovidData_clean.columns)
# myCovidData_clean.df.describe()
print("\n")
myCovidData_chaos.save_csv_data(output_file_name)

"""mystring = "id-jk"
print(mystring.upper())"""

"""myDF = myCovidData_clean.df[myCovidData_clean.df["Location ISO Code"] == "ID-JK"]
myDF_JK = myDF.copy()
myDF = myCovidData_clean.df[myCovidData_clean.df["Location ISO Code"] == "ID-JB"]
myDF_JB = myDF.copy()

print(myDF_JK)
print(myDF_JK.columns)
print(myDF_JB)
print(myDF_JB.columns)"""
print("\n")

mylist = ["ID-JB", "ID-JK", "ID-PA"]
myCovidData_clean = myCovidData_clean.get_dataframe_of_interest_based_on_string("Location ISO Code", mylist)
print(myCovidData_clean.df_oi_dict.keys())
print("\n")

myDF_JB = myCovidData_clean.df_oi_dict["Location ISO Code: ID-JB"]
print(myDF_JB)
print(myDF_JB.describe())
print("\n")
# sum_of_cases = myDF_JB["New Cases"].sum()
# print(sum_of_cases)

# myDF_JB.plot.hist()
# myDF_JB.plot.scatter(x="Date", y="New Cases")
# myDF_JB.plot.show()

print("Statistics:  ")
print("before :  ", myCovidData_clean.df_oi_statistics)
mySelectedStats = ["New Cases", "New Deaths", "New Recovered", "New Active Cases"]
myCovidData_clean = myCovidData_clean.calculate_descriptive_statistics(mySelectedStats)
print("after :  ", myCovidData_clean.df_oi_statistics)
print("\n")

myCovidData_clean.plot_from_saved_dict("Date", "New Cases", "Location ISO Code: ID-JB")
# myCovidData_clean.plot_from_saved_dict("Date", "New Cases", "Location ISO Code: ID-JK")

print(myCovidData_clean.current_directory)
print(myCovidData_clean.output_directory)


# end of file, version: 20:20 04.10.2023
