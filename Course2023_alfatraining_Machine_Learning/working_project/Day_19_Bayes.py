# Python, using Anaconda environment
# Week 4, Day 19

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

random_seed = 42
np.random.seed(random_seed)

# prepare white wine data, with pandas
wine_data = pd.read_csv("winequality-white.csv", delimiter=";")
# print(wine_data, type(wine_data))
# print(wine_data.describe())

wine_target_df = wine_data["quality"]
wine_feature_df = wine_data.drop(columns=["quality"])
# print(wine_target_df)
# print(wine_feature_df)

test_size_ratio = 0.3
wine_feature_train, wine_feature_test, wine_target_train, wine_target_test = train_test_split(wine_feature_df, wine_target_df, test_size=test_size_ratio, random_state=random_seed)

# Bayes
myBayes = GaussianNB()
myBayes.fit(wine_feature_train, wine_target_train)

bayes_test_score = myBayes.score(wine_feature_test, wine_target_test)
bayes_training_score = myBayes.score(wine_feature_train, wine_target_train)
y_pred = myBayes.predict(wine_feature_test)

print(f"naive Bayes training score:  {bayes_training_score} ")
print(f"naive Bayes test score:  {bayes_test_score} ")
print("\n")

# cv ???




# end of file
