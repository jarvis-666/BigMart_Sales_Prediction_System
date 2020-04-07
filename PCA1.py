# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:14:55 2020

@author: Vin
"""

import pandas as pd           # panel data, to manipulate data in python
import numpy as np            # allows us to generate numbers and do other mathy things

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# importing the datasets that have been pre-processed
train_data_PCA = pd.read_csv("train_modified.csv")
test_data_PCA = pd.read_csv("test_modified.csv")

# extracting the target variable and features from the training dataset
train_data_PCA_X = train_data_PCA.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis = 1)
test_data_PCA_X = test_data_PCA.drop(['Item_Identifier', 'Outlet_Identifier'], axis = 1)
train_data_PCA_y = train_data_PCA.iloc[:, 5]

# For Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('col_tran', StandardScaler(), ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years'])], remainder = 'passthrough')
train_data_PCA_X = ct.fit_transform(train_data_PCA_X)
test_data_PCA_X = ct.transform(test_data_PCA_X)

# Principal Component Analysis starts here
pca = PCA()
train_data_PCA_X = pca.fit_transform(train_data_PCA_X, train_data_PCA_y)
test_data_PCA_X = pca.transform(test_data_PCA_X)
explained_variance = pca.explained_variance_ratio_      # this contains the contribution of each principal component

# applying the models as applied before
from sklearn.linear_model import LinearRegression, Ridge
lr = LinearRegression()
lr.fit(train_data_PCA_X, train_data_PCA_y)

# predicting to find the accuracy
y_pred_train_lr = lr.predict(train_data_PCA_X)
y_pred_test_lr = lr.predict(test_data_PCA_X)

rr = Ridge(alpha = 0.05)
rr.fit(train_data_PCA_X, train_data_PCA_y)
y_pred_train_rr = rr.predict(train_data_PCA_X)
y_pred_test_rr = rr.predict(test_data_PCA_X)

# performing cross validation here
from sklearn.model_selection import cross_validate
from sklearn import metrics
cv_score = cross_validate(lr, train_data_PCA_X, train_data_PCA_y, cv = 20, scoring = 'neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score['test_score']))

cv_score_rr = cross_validate(rr, train_data_PCA_X, train_data_PCA_y, cv = 20, scoring = 'neg_mean_squared_error')
cv_score_rr = np.sqrt(np.abs(cv_score_rr['test_score']))

# Cross validation and performance metric results
with open('lr_PCA.txt', "w", encoding="utf-8") as f:
        f.write("Model Report")
        f.write(f"\nRMSE : {np.sqrt(metrics.mean_squared_error((train_data_PCA_y).values, y_pred_train_lr))}")
        f.write(f"\nCV Score: Mean-{np.mean(cv_score)} | Std-{np.std(cv_score)} | Min-{np.min(cv_score)} | Max-{np.max(cv_score)}")

print("Model Report for Linear Regression")
print(f"\nRMSE : {np.sqrt(metrics.mean_squared_error((train_data_PCA_y).values, y_pred_train_lr))}")
print(f"\nCV Score: Mean-{np.mean(cv_score)} | Std-{np.std(cv_score)} | Min-{np.min(cv_score)} | Max-{np.max(cv_score)}")

with open('rr_PCA.txt', "w", encoding="utf-8") as f:
        f.write("Model Report")
        f.write(f"\nRMSE : {np.sqrt(metrics.mean_squared_error((train_data_PCA_y).values, y_pred_train_rr))}")
        f.write(f"\nCV Score: Mean-{np.mean(cv_score_rr)} | Std-{np.std(cv_score_rr)} | Min-{np.min(cv_score_rr)} | Max-{np.max(cv_score_rr)}")

print("\nModel Report for Ridge Regression")
print(f"\nRMSE : {np.sqrt(metrics.mean_squared_error((train_data_PCA_y).values, y_pred_train_rr))}")
print(f"\nCV Score: Mean-{np.mean(cv_score_rr)} | Std-{np.std(cv_score_rr)} | Min-{np.min(cv_score_rr)} | Max-{np.max(cv_score_rr)}")
