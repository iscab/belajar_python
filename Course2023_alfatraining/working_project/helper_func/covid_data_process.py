""" Covid 19 Data Processing

this file contains classes and functions for data processing

environment: Python 3.11 under Anaconda 23.7.4

file version: 18:28 05.10.2023

"""

__author__ = "Ignatius S. Condro Atmawan"
__contact__ = "saptocondro@gmail.com"
__copyright__ = "alfatraining"
__date__ = "2023/10/05"

import os

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt


class CovidData:
    """Class to process Covid 19 Data"""
    def __init__(self):
        """create class object: Covid Data"""
        self.about = "Process Covid Data"
        self.file_name = ""  # the name of the file to be analyzed
        self.current_directory = ""
        self.output_directory = ""
        self.df = None  # pandas data frame
        self.columns = []  # list of columns of the data frame
        self.df_oi_dict = {}  # data frame of interest
        self.df_oi_statistics = {}  # descriptive statistics from data frame of interest
        self.assign_directory()

    def __str__(self):
        """About the Covid Data object"""
        return self.about

    def assign_directory(self):
        """
        Get current directory/folder and then prepare output directory for result

        :return: Covid Data object itself
        """
        self.current_directory = os.getcwd()
        self.output_directory = os.path.join(self.current_directory, "results")

        # make output directory
        if not os.path.isdir(self.output_directory):
            os.mkdir(self.output_directory)
        return self

    def read_csv_data(self, file_name: str):
        """
        Read data from csv file

        :param file_name: (str) file name
        :return: Covid Data object itself
        """
        self.file_name = file_name
        self.df = pd.read_csv(self.file_name)

        self.columns = list(self.df.columns)
        return self

    def save_csv_data(self, file_name: str):
        """
        Save data into csv file

        :param file_name: (str) file name
        :return: None
        """
        self.df.to_csv(file_name)

    def select_column(self, list_of_columns: list):
        """
        Select data based of column names

        :param list_of_columns: list of column that we want to select
        :return: Covid Data object itself
        """
        self.df = self.df.filter(items=list_of_columns)
        self.columns = list(self.df.columns)
        # print(self.df)
        return self

    def omit_empty_data(self):
        """
        Omit/Drop empty data, e.g. NaN or None

        :return: Covid Data object itself
        """
        # omit NaN
        self.df = self.df.dropna()
        self.columns = list(self.df.columns)
        # print(self.df)
        return self

    def clean_numeric_data(self, list_of_columns: list):
        """
        Clean the numeric data by removing/dropping non numeric elements

        :param list_of_columns: (list) containing the variables of interest
        :return: Covid Data object itself
        """
        for idx, column_oi in enumerate(list_of_columns):
            # print(column_oi)
            self.df[column_oi] = pd.to_numeric(self.df[column_oi], errors="coerce")
            self.df = self.df.dropna(subset=[column_oi])
        return self

    def get_dataframe_of_interest_based_on_string(self, column_name: str, values: list):
        """
        Select variable of interest by the column name and the values that we want.

        :param column_name: (str) variable of interest
        :param values: (list) contain the values of interest inside the column/variable
        :return: Covid Data object itself
        """
        for idx, df_value_oi in enumerate(values):
            # print(idx, " : ", df_value_oi)
            df_oi = self.df[self.df[column_name] == df_value_oi]
            # print(df_oi)
            dict_key = column_name + ": " + df_value_oi
            self.df_oi_dict[dict_key] = df_oi
        # print(self.df_oi_dict)
        return self

    def plot_from_saved_dict(self, x_col: str, y_col: str, dict_key: str, need_show: bool):
        """
        Visualize the data

        :param x_col: (str) column/variable names for x axis, e.g. Date
        :param y_col: (str) column/variable names for y axis
        :param dict_key: (str) the title for the plot
        :param need_show: (bool) setting if we need to show the plot or not. True means we show the plot.
        :return:
        """
        standard_date_time_format = "covid19_graphic_%Y_%m_%d_%H_%M_%S"
        ticktime = dt.datetime.now()
        file_name_stamp = ticktime.strftime(standard_date_time_format)

        if dict_key in self.df_oi_dict.keys():
            df_oi = self.df_oi_dict[dict_key]
            # plotting
            # print(df_oi.columns)
            if x_col in df_oi.columns and y_col in df_oi.columns:
                print(f"Plotting {dict_key}")
                # create figure & plot
                plt.figure()
                plt.plot(df_oi[x_col], df_oi[y_col])
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(dict_key)

                # prepare file name
                file_name = dict_key.lower()
                file_name = file_name.replace(" ", "_")
                file_name = file_name.replace(":", "_")
                file_name = file_name.replace("-", "_")
                file_name = file_name.replace(".", "_")
                file_name = file_name_stamp + "_" + file_name
                # print(file_name)

                # save the figure
                png_file_name = file_name + ".png"
                png_file_name = os.path.join(self.output_directory, png_file_name)
                plt.savefig(png_file_name)

                if need_show:
                    plt.show()
            else:
                print("We don't have that kind of data for plotting")
        else:
            print("We don't have that kind of data for plotting")

    def calculate_descriptive_statistics(self, list_of_columns: list):
        """
        Calculate the descriptive statistics of the variables/columns from the data frame of interest.

        :param list_of_columns: (list) containing the selected variables/columns
        :return: Covid Data object itself
        """
        for key, df_oi in self.df_oi_dict.items():
            # print(key)
            # print(key, " :  ", df_oi.columns)
            stat_dict = {}
            for idx, column_name in enumerate(list_of_columns):
                stat_dict_core = {}
                if column_name in df_oi:
                    # print(f"{column_name} is here in {key}")
                    stat_dict_core["sum"] = df_oi[column_name].sum()
                    stat_dict_core["mean"] = df_oi[column_name].mean()
                    stat_dict_core["median"] = df_oi[column_name].median()
                    # summarize statistics
                    stat_dict[column_name] = stat_dict_core
                else:
                    print("We don't have that kind of data for statistics calculation")
            # print(stat_dict)
            self.df_oi_statistics[key] = stat_dict

        return self

    def save_descriptive_statistics(self):
        """
        Save the descriptive statistics into a text file

        :return: None
        """
        standard_date_time_format = "covid19_statistics_%Y_%m_%d_%H_%M_%S"
        ticktime = dt.datetime.now()
        file_name_stamp = ticktime.strftime(standard_date_time_format)
        # print(file_name_stamp)

        for stat_key, stat_dict in self.df_oi_statistics.items():
            # print(stat_key)
            # print(stat_dict)
            # Prepare file name
            file_name = stat_key.lower()
            file_name = file_name.replace(" ", "_")
            file_name = file_name.replace(":", "_")
            file_name = file_name.replace("-", "_")
            file_name = file_name.replace(".", "_")
            # print(file_name)

            # Prepare file content
            file_content = stat_key + "\n"
            file_content += "data sets:  " + self.file_name + "\n\n"
            for column_oi, stats_oi in stat_dict.items():
                # print(column_oi)
                # print(stats_oi)
                file_content += column_oi + "\n"
                for key, stat_value in stats_oi.items():
                    file_content += f"{key} =  {stat_value}\n"
                file_content += "\n"
            # print(file_content)

            # Saving file
            txt_file_name = file_name_stamp + "_" + file_name + ".txt"
            txt_file_name = os.path.join(self.output_directory, txt_file_name)
            # print(txt_file_name)
            try:
                open(txt_file_name, "w", encoding="utf-8").write(file_content)
            except RuntimeError:
                print("error saving file")





# read this:  https://matplotlib.org/stable/tutorials/pyplot.html
# read this:  https://pandas.pydata.org/
# read this:  https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
# read this:  https://peps.python.org/pep-0257/
# read this:  https://gist.github.com/NicolasBizzozzero/6d4ca63f8482a1af99b0ed022c13b041
# read this:  https://www.datacamp.com/tutorial/docstrings-python

# end of file, version: 18:28 05.10.2023
