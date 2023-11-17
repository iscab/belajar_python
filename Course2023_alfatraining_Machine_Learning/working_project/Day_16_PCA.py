# Python, using Anaconda environment
# Week 4, Day 16

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

random_seed = 42
np.random.seed(random_seed)

# prepare wine data, with pandas
df_alle = pd.read_pickle("df_alle_2023_10_30.pkl")
# print(df_alle, type(df_alle))
print(df_alle.describe())
print("\n")

wine_target_df = df_alle["quality"]
wine_feature_df = df_alle.drop(columns=["quality"])
# print(wine_target_df)
# print(wine_feature_df)

test_size_ratio = 0.3
wine_feature_train, wine_feature_test, wine_target_train, wine_target_test = train_test_split(wine_feature_df, wine_target_df, test_size=test_size_ratio, random_state=random_seed)

# PCA no scaling
pca_no_scale = PCA()
wine_transform = pca_no_scale.fit_transform(wine_feature_train)

print(f"explained variance ratio:  {pca_no_scale.explained_variance_ratio_} ")
print(f"explained variance:  {pca_no_scale.explained_variance_} ")
print(pca_no_scale.feature_names_in_)
print(pca_no_scale.get_feature_names_out())
# print(df_alle.columns)
print("\n")
# print("components: \n", pca_no_scale.components_)
# print("components in absolute: \n ", abs(pca_no_scale.components_))
print("\n")

# PCA with scaling
scaler = StandardScaler()
# scaler.fit(wine_feature_train)
wine_z_scale = scaler.fit_transform(wine_feature_train)
print(type(wine_feature_train), type(wine_z_scale))

pca_scale = PCA()
wine_transform_z = pca_scale.fit_transform(wine_z_scale)

print(f"explained variance ratio:  {pca_scale.explained_variance_ratio_} ")
print(f"explained variance:  {pca_scale.explained_variance_} ")
print(pca_scale.get_feature_names_out())
# print(df_alle.columns)
print("\n")



# end of file
