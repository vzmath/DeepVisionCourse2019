import time
import pandas as pd
import numpy as np

import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from utils import *

filename_mushrooms = 'mushrooms.csv'
df_mushrooms = pd.read_csv(filename_mushrooms)

# data preprocessing
#############################################################################################################################################
for col in df_mushrooms.columns.values:
    print(col, df_mushrooms[col].unique())
# Remove columns with only 1 value
for col in df_mushrooms.columns.values:
    if len(df_mushrooms[col].unique()) <= 1:
        print("Removing column {}, which only contains the value: {}".format(col, df_mushrooms[col].unique()[0]))

# Drop rows with missing values
print("Number of rows in total: {}".format(df_mushrooms.shape[0]))
print("Number of rows with missing values in column 'stalk-root': {}".format(df_mushrooms[df_mushrooms['stalk-root'] == '?'].shape[0]))
df_mushrooms_dropped_rows = df_mushrooms[df_mushrooms['stalk-root'] != '?']
# Drop column with more than X percent missing values
drop_percentage = 0.8

df_mushrooms_dropped_cols = df_mushrooms.copy(deep=True)
df_mushrooms_dropped_cols.loc[df_mushrooms_dropped_cols['stalk-root'] == '?', 'stalk-root'] = np.nan

for col in df_mushrooms_dropped_cols.columns.values:
    no_rows = df_mushrooms_dropped_cols[col].isnull().sum()
    percentage = no_rows / df_mushrooms_dropped_cols.shape[0]
    if percentage > drop_percentage:
        del df_mushrooms_dropped_cols[col]
        print("Column {} contains {} missing values. This is {} percent. Dropping this column.".format(col, no_rows, percentage))

# Fill missing values with zero / -1
f_mushrooms_zerofill = df_mushrooms.copy(deep = True)
df_mushrooms_zerofill.loc[df_mushrooms_zerofill['stalk-root'] == '?', 'stalk-root'] = np.nan
df_mushrooms_zerofill.fillna(0, inplace=True)

# Fill missing values with backward fill
df_mushrooms_bfill = df_mushrooms.copy(deep = True)
df_mushrooms_bfill.loc[df_mushrooms_bfill['stalk-root'] == '?', 'stalk-root'] = np.nan
df_mushrooms_bfill.fillna(method='bfill', inplace=True)

# Fill missing values with forward fill
df_mushrooms_ffill = df_mushrooms.copy(deep = True)
df_mushrooms_ffill.loc[df_mushrooms_ffill['stalk-root'] == '?', 'stalk-root'] = np.nan
df_mushrooms_ffill.fillna(method='ffill', inplace=True)

# One-Hot encoding the columns with categorical data
df_mushrooms_ohe = df_mushrooms.copy(deep=True)
to_be_encoded_cols = df_mushrooms_ohe.columns.values
label_encode(df_mushrooms_ohe, to_be_encoded_cols)

df_mushrooms_dropped_rows_ohe = df_mushrooms_dropped_rows.copy(deep = True)
df_mushrooms_zerofill_ohe = df_mushrooms_zerofill.copy(deep = True)
df_mushrooms_bfill_ohe = df_mushrooms_bfill.copy(deep = True)
df_mushrooms_ffill_ohe = df_mushrooms_ffill.copy(deep = True)

label_encode(df_mushrooms_dropped_rows_ohe, to_be_encoded_cols)
label_encode(df_mushrooms_zerofill_ohe, to_be_encoded_cols)
label_encode(df_mushrooms_bfill_ohe, to_be_encoded_cols)
label_encode(df_mushrooms_ffill_ohe, to_be_encoded_cols)

# Expanding the columns with categorical data
y_col = 'class'
to_be_expanded_cols = list(df_mushrooms.columns.values)
to_be_expanded_cols.remove(y_col)

df_mushrooms_expanded = df_mushrooms.copy(deep=True)
label_encode(df_mushrooms_expanded, [y_col])
expand_columns(df_mushrooms_expanded, to_be_expanded_cols)
#############################################################################################################################################

# classification
#############################################################################################################################################
dict_dataframes = {
    "df_mushrooms_ohe": df_mushrooms_ohe,
    "df_mushrooms_dropped_rows_ohe": df_mushrooms_dropped_rows_ohe,
    "df_mushrooms_zerofill_ohe": df_mushrooms_zerofill_ohe,
    "df_mushrooms_bfill_ohe": df_mushrooms_bfill_ohe,
    "df_mushrooms_ffill_ohe": df_mushrooms_ffill_ohe,
    "df_mushrooms_expanded": df_mushrooms_expanded,
    "df_mushrooms_dropped_rows_expanded": df_mushrooms_dropped_rows_expanded,
    "df_mushrooms_zerofill_expanded": df_mushrooms_zerofill_expanded,
    "df_mushrooms_bfill_expanded": df_mushrooms_bfill_expanded,
    "df_mushrooms_ffill_expanded": df_mushrooms_ffill_expanded
}

y_col = 'class'
train_test_ratio = 0.7

for df_key, df in dict_dataframes.items():
    x_cols = list(df.columns.values)
    x_cols.remove(y_col)
    df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df, y_col, x_cols, train_test_ratio)
    dict_models = batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 8, verbose=False)

    print()
    print(df_key)
    display_dict_models(dict_models)
    print("-------------------------------------------------------")


#############################################################################################################################################

# hyper parameter optimization
GDB_params = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.5, 0.1, 0.01, 0.001],
    'criterion': ['friedman_mse', 'mse', 'mae']
}

df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df_glass, y_col_glass, x_cols_glass, 0.6)

for n_est in GDB_params['n_estimators']:
    for lr in GDB_params['learning_rate']:
        for crit in GDB_params['criterion']:
            clf = GradientBoostingClassifier(n_estimators=n_est,
                                             learning_rate = lr,
                                             criterion = crit)
            clf.fit(X_train, Y_train)
            train_score = clf.score(X_train, Y_train)
            test_score = clf.score(X_test, Y_test)
            print("For ({}, {}, {}) - train, test score: \t {:.5f} \t-\t {:.5f}".format(n_est, lr, crit[:4], train_score, test_score))
