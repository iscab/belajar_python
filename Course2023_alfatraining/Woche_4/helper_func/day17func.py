# Python, using Anaconda environment
# Week 4, Day 17

# read this:  https://matplotlib.org/stable/tutorials/pyplot.html
# read this:  https://pandas.pydata.org/
# read this:  https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf


import os

import pandas as pd
import matplotlib.pyplot as plt


class CovidData:
    """Class to process Covid 19 Data"""
    def __init__(self):
        """create class object: Covid Data"""
        self.about = "Process Covid Data"
        self.file_name = ""
        self.current_directory = ""
        self.output_directory = ""
        self.df = None  # pandas data frame
        self.columns = []
        self.df_oi_dict = {}
        self.df_oi_statistics = {}
        self.assign_directory()

    def __str__(self):
        """About the Covid Data object"""
        return self.about

    def assign_directory(self):
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
        :return: Covid Data object
        """
        self.file_name = file_name
        self.df = pd.read_csv(self.file_name)

        self.columns = list(self.df.columns)
        return self

    def save_csv_data(self, file_name: str):
        """"""
        self.df.to_csv(file_name)

    def select_column(self, list_of_columns: list):
        """
        Select data based of column names

        :param list_of_columns: list of column that we want to select
        :return: Covid Data Object
        """
        self.df = self.df.filter(items=list_of_columns)
        self.columns = list(self.df.columns)
        # print(self.df)
        return self

    def omit_empty_data(self):
        """"""
        # omit NaN
        self.df = self.df.dropna()
        self.columns = list(self.df.columns)
        # print(self.df)
        return self

    def get_dataframe_of_interest_based_on_string(self, column_name: str, values: list):
        """"""
        for idx, df_value_oi in enumerate(values):
            # print(idx, " : ", df_value_oi)
            df_oi = self.df[self.df[column_name] == df_value_oi]
            # print(df_oi)
            dict_key = column_name + ": " + df_value_oi
            self.df_oi_dict[dict_key] = df_oi
        # print(self.df_oi_dict)
        return self

    def show_plot_from_saved_dict(self, x_col: str, y_col: str, dict_key: str):
        """"""
        if dict_key in self.df_oi_dict.keys():
            df_oi = self.df_oi_dict[dict_key]
            # plotting
            # print(df_oi.columns)
            if x_col in df_oi.columns and y_col in df_oi.columns:
                print(f"Plotting {dict_key}")
                plt.plot(df_oi[x_col], df_oi[y_col])
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(dict_key)

                FileName = dict_key.replace(" ", "_").lower()
                FileName = FileName.replace(":", "")
                print(FileName)

                plt.show()
            else:
                print("We don't have that kind of data for plotting")
        else:
            print("We don't have that kind of data for plotting")

    def calculate_descriptive_statistics(self, list_of_columns: list):
        """"""
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














