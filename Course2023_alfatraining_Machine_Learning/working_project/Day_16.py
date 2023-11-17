# Python, using Anaconda environment
# Week 4, Day 16

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# prepare wine data, with pandas
red_wine_data = pd.read_csv("winequality-red.csv", delimiter=";")
white_wine_data = pd.read_csv("winequality-white.csv", delimiter=";")
# print(red_wine_data, type(red_wine_data))
# print(white_wine_data, type(white_wine_data))
red_wine_output = red_wine_data["quality"]
red_wine_feature = red_wine_data.drop(columns=["quality"])
white_wine_output = white_wine_data["quality"]
white_wine_feature = white_wine_data.drop(columns=["quality"])
# print(red_wine_output, type(red_wine_output))
# print(red_wine_feature, type(red_wine_feature))
# print(white_wine_output, type(white_wine_output))
# print(white_wine_feature, type(white_wine_feature))

red_corr = red_wine_data.corr()
print(red_corr)

test_size_ratio = 0.2
red_wine_feature_train, red_wine_feature_test, red_wine_output_train, red_wine_output_test = train_test_split(red_wine_feature, red_wine_output, test_size=test_size_ratio)
white_wine_feature_train, white_wine_feature_test, white_wine_output_train, white_wine_output_test = train_test_split(white_wine_feature, white_wine_output, test_size=test_size_ratio)
# print(red_wine_feature_train, type(red_wine_feature_train))
# print(white_wine_feature_train, type(white_wine_feature_train))


# end of file
