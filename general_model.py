# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:25:07 2020

@author: Vin
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Creating a function that takes in the dataset, model and calculates prediction, cross validation score and RMSE

# importing the final datasets that have already been preprocessed
train_df = pd.read_csv('train_modified.csv')
test_df = pd.read_csv('test_modified.csv')

# defining target and ID columns
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier', 'Outlet_Identifier']

from sklearn.model_selection import cross_validate
from sklearn import metrics

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename1, filename2):
    # fit the algorithm on the dataset
    alg.fit(dtrain[predictors], dtrain[target])
    
    # predict training set
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    # perform cross-validation
    cv_score = cross_validate(alg, dtrain[predictors], dtrain[target], cv = 20, scoring = 'neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score['test_score']))
    
    # predict on testing data
    dtest[target] = alg.predict(dtest[predictors])
    
    # export submission file
    IDcol.append(target)
    submission = pd.DataFrame({x: dtest[x] for x in IDcol})
    submission.to_csv(filename1, index = False)
    
    with open(filename2, "w", encoding="utf-8") as f:
        f.write("Model Report")
        f.write(f"\nRMSE : {np.sqrt(metrics.mean_squared_error((dtrain[target]).values, dtrain_predictions))}")
        f.write(f"\nCV Score: Mean-{np.mean(cv_score)} | Std-{np.std(cv_score)} | Min-{np.min(cv_score)} | Max-{np.max(cv_score)}")
    
    print("\nModel Report")
    print(f"\nRMSE : {np.sqrt(metrics.mean_squared_error((dtrain[target]).values, dtrain_predictions))}")
    print(f"\nCV Score: Mean-{np.mean(cv_score)} | Std-{np.std(cv_score)} | Min-{np.min(cv_score)} | Max-{np.max(cv_score)}")
