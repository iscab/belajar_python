# Python, using Anaconda environment
# Week 4, Day 19

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

random_seed = 42
np.random.seed(random_seed)

# prepare white wine data, with pandas
wine_data = pd.read_csv("winequality-white.csv", delimiter=";")
# print(wine_data, type(wine_data))
# print(wine_data.describe())

# transform wine quality into 3 categories
bins = [0, 4, 6, 10]  # Specify the bin edges
labels = [1, 2, 3]    # Specify the labels for each category
wine_data["new_quality"] = pd.cut(wine_data["quality"], bins=bins, labels=labels, include_lowest=True)

wine_output = wine_data["new_quality"]
wine_feature = wine_data.drop(columns=["quality", "new_quality"])
# print(wine_output, type(wine_output))
# print(wine_feature, type(wine_feature))


wine_corr = wine_data.corr()
# print(wine_corr)
print("\n")

# Z-scaled
z_scale = StandardScaler()
wine_feature_scaled = z_scale.fit_transform(wine_feature)

pca = PCA()
wine_feature_PCA = pca.fit_transform(wine_feature_scaled)

# select best eigen values
print("explained variance ratio:  ", pca.explained_variance_ratio_)
# kumulierte Summe von explained_variance_ratio_
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
print("cumulative explained variance:  ", cumulative_explained_variance)

# selected components
pca_new = PCA(n_components=8)
wine_feature_PCA_new = pca_new.fit_transform(wine_feature_scaled)

# train and test split
test_size_ratio = 0.3
wine_feature_train, wine_feature_test, wine_output_train, wine_output_test = train_test_split(wine_feature_PCA_new, wine_output, test_size=test_size_ratio, random_state=random_seed)

print("test output data: ", wine_output_test.describe(), "\n")
print("train output data: ", wine_output_train.describe(), "\n")
print("\n")

# logistic regression model
log_reg = LogisticRegression(max_iter=50_000)
ticktime = dt.datetime.now()

log_reg.fit(wine_feature_train, wine_output_train)
log_reg_duration = dt.datetime.now() - ticktime

log_reg_train_score = log_reg.score(wine_feature_train, wine_output_train)
log_reg_test_score = log_reg.score(wine_feature_test, wine_output_test)

print(f"logistic regression training score:  {log_reg_train_score} ")
print(f"logistic regression test score:  {log_reg_test_score} ")

print("\n")

wine_output_pred = log_reg.predict(wine_feature_test)
# print(wine_output_pred, type(wine_output_pred))

# cross validation
log_reg_cross_val_train_score = cross_val_score(log_reg, wine_feature_train, wine_output_train, cv=7)
print("logistic regression cross-validation training score: \n", log_reg_cross_val_train_score, type(log_reg_cross_val_train_score))
print("mean:  ", log_reg_cross_val_train_score.mean())
print("standard deviation:  ", log_reg_cross_val_train_score.std())
print("\n")
# print(log_reg_cross_val_train_score.min())

log_reg_cross_val_train_score_1 = cross_validate(log_reg, wine_feature_train, wine_output_train, cv=7)
print(log_reg_cross_val_train_score_1)
print("\n")


# classification_report

# grid search with cross validation
parameters = [
    {"solver": ["lbfgs"], "penalty": ["l2"], "multi_class": ["ovr", "multinomial"]},
    {"solver": ["saga"], "penalty": ["l2", "l1"], "multi_class": ["ovr", "multinomial"]}
]

log_reg_grid = GridSearchCV(log_reg, parameters, cv=7)
fitting = log_reg_grid.fit(wine_feature_train, wine_output_train)
print(fitting)
print(log_reg_grid.cv_results_)
print("\n")

best_log_reg = log_reg_grid.best_estimator_
print(best_log_reg)

best_training_score = log_reg_grid.score(wine_feature_train, wine_output_train)
best_test_score = log_reg_grid.score(wine_feature_test, wine_output_test)
print("best training score:  ", best_training_score)
print("best test score:  ", best_test_score)




# end of file
