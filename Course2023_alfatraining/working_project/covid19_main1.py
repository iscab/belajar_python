""" Covid 19 data processing

Procedure of data processing:
1. select variables of interest, from the data
2. clean the data
3. calculate descriptive statistics
4. visualize the data with diagrams or other images

data sources:  https://www.kaggle.com/datasets/hendratno/covid19-indonesia/data

environment: Python 3.11 under Anaconda 23.7.4

file version: 18:28 05.10.2023

"""

__author__ = "Ignatius S. Condro Atmawan"
__contact__ = "saptocondro@gmail.com"
__copyright__ = "alfatraining"
__date__ = "2023/10/05"

import os
import time
import pandas as pd
import matplotlib.pyplot as plt

import helper_func.covid_data_process as cvd

# create object: Covid Data
myCovidData_clean = cvd.CovidData()
myCovidData_chaos = cvd.CovidData()
print("\n")


# prepare file name
data_sets_folder = "data_sets"
FileName_clean = "covid_19_indonesia_time_series_all.csv"
FileName_chaos = "Ignatius_covid_19_indonesia_time_series_all_chaos.csv"
# print(myCovidData_clean.output_directory)
# print(myCovidData_chaos.output_directory)
FileName_clean = os.path.join(data_sets_folder, FileName_clean)
FileName_clean = os.path.join(myCovidData_clean.current_directory, FileName_clean)
print("clean data:  ", FileName_clean)
FileName_chaos = os.path.join(data_sets_folder, FileName_chaos)
FileName_chaos = os.path.join(myCovidData_chaos.current_directory, FileName_chaos)
print("chaos data:  ", FileName_chaos)
print("\n")

# read the CSV file
myCovidData_clean = myCovidData_clean.read_csv_data(FileName_clean)
# print(myCovidData_clean.df)
# print(myCovidData_clean.df.columns)
print(myCovidData_clean.columns)
# print("\n")
myCovidData_chaos = myCovidData_chaos.read_csv_data(FileName_chaos)
# print(myCovidData_chaos.df)
# print(myCovidData_chaos.df.columns)
print(myCovidData_chaos.columns)
print("\n")


# Selecting variables of interest
# mySelectedColumns = ["Date", "Location ISO Code", "Location", "New Cases", "New Deaths", "New Recovered", "New Active Cases", "Longitude", "Latitude"]
mySelectedColumns_chaos = ["Date", "Location ISO Code", "Location", "New Cases.1", "New Deaths.1", "New Recovered.1", "New Active Cases.1"]
columns_with_expected_integer_chaos = ["New Cases.1", "New Deaths.1", "New Recovered.1", "New Active Cases.1"]
mySelectedColumns_clean = ["Date", "Location ISO Code", "Location", "New Cases", "New Deaths", "New Recovered", "New Active Cases"]
columns_with_expected_integer_clean = ["New Cases", "New Deaths", "New Recovered", "New Active Cases"]

# chaos data sets
myCovidData_chaos = myCovidData_chaos.select_column(mySelectedColumns_chaos)
myCovidData_chaos = myCovidData_chaos.clean_numeric_data(columns_with_expected_integer_chaos)
myCovidData_chaos = myCovidData_chaos.omit_empty_data()
# print(myCovidData_chaos.df)
# print(myCovidData_chaos.df.columns)
# print(myCovidData_chaos.columns)
# print("\n")
output_file_name = "selected_covid19_chaos_data.csv"
output_file_name = os.path.join(myCovidData_chaos.output_directory, output_file_name)
myCovidData_chaos.save_csv_data(output_file_name)

# clean data sets
myCovidData_clean = myCovidData_clean.select_column(mySelectedColumns_clean)
myCovidData_clean = myCovidData_clean.clean_numeric_data(columns_with_expected_integer_clean)
myCovidData_clean = myCovidData_clean.omit_empty_data()
# print(myCovidData_clean.df)
# print(myCovidData_clean.df.columns)
# print(myCovidData_clean.columns)
# myCovidData_clean.df.describe()
# print("\n")
output_file_name = "selected_covid19_clean_data.csv"
output_file_name = os.path.join(myCovidData_clean.output_directory, output_file_name)
myCovidData_clean.save_csv_data(output_file_name)
# TODO: develop cleaning methods for chaos data


# processing the clean data sets: statistics and visualization
mylist = ["ID-JB", "ID-JK", "ID-PA"]
myCovidData_clean = myCovidData_clean.get_dataframe_of_interest_based_on_string("Location ISO Code", mylist)
print(myCovidData_clean.df_oi_dict.keys())
print("\n")

mylist = ["Jawa Barat", "DKI Jakarta", "Papua"]
myCovidData_clean = myCovidData_clean.get_dataframe_of_interest_based_on_string("Location", mylist)
print(myCovidData_clean.df_oi_dict.keys())
print("\n")

print("Statistics:  ")
mySelectedStats = ["New Cases", "New Deaths", "New Recovered", "New Active Cases"]
print("before :  ", myCovidData_clean.df_oi_statistics)
myCovidData_clean = myCovidData_clean.calculate_descriptive_statistics(mySelectedStats)
print("after :  ", myCovidData_clean.df_oi_statistics)
print("\n")
myCovidData_clean.save_descriptive_statistics()

# plot figures
myCovidData_clean.plot_from_saved_dict("Date", "New Cases", "Location: Jawa Barat", True)
myCovidData_clean.plot_from_saved_dict("Date", "New Cases", "Location: Papua", False)
myCovidData_clean.plot_from_saved_dict("Date", "New Cases", "Location: DKI Jakarta", False)

# TODO: develop cleaning methods for chaos data

# processing the chaos data sets: data cleaning
"""myDF = myCovidData_chaos.df.head(7)
# myDF = myCovidData_clean.df.head(7)
print(myDF)
# self.df.to_csv(file_name)
myDF.to_csv("dummy_df.csv")
# print(myDF.dtypes)
for index, row in myDF.iterrows():
    # print(row["Date"], type(row["Date"]))
    print(row["New Cases"], type(row["New Cases"]))"""

"""myDF = myCovidData_chaos.df
print(myDF)
myDF["New Cases"] = pd.to_numeric(myDF["New Cases"], errors="coerce")
myDF = myDF.dropna(subset=["New Cases"])
print(myDF)"""



# processing the chaos data sets: statistics and visualization
mylist = ["ID-JB", "ID-JK", "ID-PA"]
myCovidData_chaos = myCovidData_chaos.get_dataframe_of_interest_based_on_string("Location ISO Code", mylist)
print(myCovidData_chaos.df_oi_dict.keys())
print("\n")

mylist = ["Jawa Barat", "DKI Jakarta", "Papua"]
myCovidData_chaos = myCovidData_chaos.get_dataframe_of_interest_based_on_string("Location", mylist)
print(myCovidData_chaos.df_oi_dict.keys())
print("\n")

print("Statistics:  ")
mySelectedStats = ["New Cases.1", "New Deaths.1", "New Recovered.1", "New Active Cases.1"]
print("before :  ", myCovidData_chaos.df_oi_statistics)
myCovidData_chaos = myCovidData_chaos.calculate_descriptive_statistics(mySelectedStats)
print("after :  ", myCovidData_chaos.df_oi_statistics)
print("\n")
myCovidData_chaos.save_descriptive_statistics()

# plot figures
myCovidData_chaos.plot_from_saved_dict("Date", "New Cases.1", "Location: Jawa Barat", True)
myCovidData_chaos.plot_from_saved_dict("Date", "New Cases.1", "Location: Papua", False)
myCovidData_chaos.plot_from_saved_dict("Date", "New Cases.1", "Location: DKI Jakarta", False)





# end of file, version: 18:28 05.10.2023
