# Python, using Anaconda environment
# Week 4, Day 16

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

random_seed = 42
np.random.seed(random_seed)

# prepare wine data, with pandas
df_alle = pd.read_pickle("df_alle_2023_10_30.pkl")
# print(df_alle, type(df_alle))
print(df_alle.describe())
print("\n")

# transform wine quality into 3 categories
bins = [0, 4, 6, 10]  # Specify the bin edges
labels = [1, 2, 3]    # Specify the labels for each category
df_alle["new_quality"] = pd.cut(df_alle["quality"], bins=bins, labels=labels, include_lowest=True)

wine_target_df = df_alle["new_quality"]
wine_feature_df = df_alle.drop(columns=["quality", "new_quality"])
# print(wine_target_df)
# print(wine_feature_df)

test_size_ratio = 0.3
wine_feature_train, wine_feature_test, wine_target_train, wine_target_test = train_test_split(wine_feature_df, wine_target_df, test_size=test_size_ratio, random_state=random_seed)

# Bayes
myBayes = CategoricalNB()
myBayes.fit(wine_feature_train, wine_target_train)

# bayes_test_score = myBayes.score(wine_feature_test, wine_target_test)  # IndexError: index 2 is out of bounds for axis 1 with size 2
bayes_training_score = myBayes.score(wine_feature_train, wine_target_train)
y_pred = myBayes.predict(wine_feature_test)
print(y_pred, type(y_pred))

print(f"naive Bayes training score:  {bayes_training_score} ")
# print(f"naive Bayes test score:  {bayes_test_score} ")
print("\n")


# Traceback (most recent call last):
#   File "C:\Users\alfa\PycharmProjects\machine_learning\ML\working_project\Day_16_Bayes.py", line 33, in <module>
#     bayes_test_score = myBayes.score(wine_feature_test, wine_target_test)
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\alfa\.conda\envs\ML\Lib\site-packages\sklearn\base.py", line 668, in score
#     return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
#                              ^^^^^^^^^^^^^^^
#   File "C:\Users\alfa\.conda\envs\ML\Lib\site-packages\sklearn\naive_bayes.py", line 106, in predict
#     jll = self._joint_log_likelihood(X)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\alfa\.conda\envs\ML\Lib\site-packages\sklearn\naive_bayes.py", line 1530, in _joint_log_likelihood
#     jll += self.feature_log_prob_[i][:, indices].T
#            ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
# IndexError: index 2 is out of bounds for axis 1 with size 2



# end of file

