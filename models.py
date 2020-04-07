# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:36:08 2020

@author: Vin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import general_model as gm

from sklearn.linear_model import LinearRegression, Ridge
# performing multiple Linear Regression
lr = LinearRegression(normalize = True)

predictors = gm.train_df.columns.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'])
gm.modelfit(lr, gm.train_df, gm.test_df, predictors, gm.target, gm.IDcol, 'LR.csv', 'LR.txt')

coef1 = pd.Series(lr.coef_, predictors).sort_values()
plt.figure()
coef1.plot(kind = 'bar', title = 'Model Coefficients for Linear Regression')
plt.show()

# performing Ridge Regression
rr = Ridge(alpha = 0.05, normalize = True)

gm.modelfit(rr, gm.train_df, gm.test_df, predictors, gm.target, gm.IDcol, 'RR.csv', 'RR.txt')

coef2 = pd.Series(rr.coef_, predictors).sort_values()
plt.figure()
coef2.plot(kind = 'bar', title = 'Model Coefficients for Ridge Regression')
plt.show()