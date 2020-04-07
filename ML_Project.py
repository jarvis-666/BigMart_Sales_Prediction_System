# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 23:08:33 2020

@author: Vin
"""

# importing the dataset
import pandas as pd       # for making the dataframe from the dataset given
import numpy as np        # for all scientific computing purposes
import matplotlib.pyplot as plt   # for creating plots and graphs
import seaborn as sns     # for creating statistical graphics

train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv("Test.csv")

first_10rows = train_data.head(n = 10) # for quickly testing if our object has the right type of data in it
print(first_10rows)

# getting the general summary of the training dataset
# Piping output of train_data.info to buffer instead of sys.stdout, to a text file
import io
buffer = io.StringIO()
train_data.info(buf = buffer)
s = buffer.getvalue()
with open("train_info.txt", "w", encoding="utf-8") as f:  
    f.write(s)


# Generating the descriptive statistics about the training dataset
# this excludes the NaN values and includes all the numeric columns
train_data_stats = train_data.describe()
print(train_data_stats)

################### EXPLORATORY DATA ANALYSIS #####################
# analysing the existent features
# check for duplicates
idsUnique = len(set(train_data.Item_Identifier))    # set data type only stores unique values
idsTotal = train_data.shape[0]
idsDupli = idsTotal - idsUnique
print(f"There are {idsDupli} duplicate IDs out of {idsTotal} total entries")

# Histogram displaying the distribution of target variable
plt.figure()
sns.distplot(train_data.Item_Outlet_Sales, bins = 25)
plt.xlabel("Item Outlet Sales")
plt.ylabel("Number of Sales")
plt.title("Item Outlet Sales Distribution")
plt.show()
print("Skew is:", train_data.Item_Outlet_Sales.skew())
print("Kurtosis is:", train_data.Item_Outlet_Sales.kurtosis())

# seeing which out of the 12 given features are numeric
numeric_features = train_data.select_dtypes(include = [np.number])
print(numeric_features)
datatypes_numeric = numeric_features.dtypes
print(datatypes_numeric)

# calculating the correlation between Numerical Predictors and Target Variable
corr = numeric_features.corr()
corr_with_target = corr["Item_Outlet_Sales"].sort_values(ascending = False)
print(corr)
print(corr_with_target)

# creating a correlation matrix using heatmap
plt.figure()
sns.heatmap(corr, vmax = 0.8)   # vmax used to anchor the colormap
plt.show()

# univariate analysis for categorical predictors
categorical_features = train_data.select_dtypes(exclude = [np.number])
print(categorical_features)

# displaying bar charts and frequency counts for categorical features
plt.figure()
sns.countplot(train_data.Item_Fat_Content) ###### MUST BE CORRECTED ######
plt.title("Item_Fat_Content")
plt.show()

plt.figure()
sns.countplot(train_data.Item_Type)
plt.xticks(rotation = 90)
plt.title("Item_Type")
plt.show()

plt.figure()
sns.countplot(train_data.Outlet_Size)
plt.title("Outlet_Size")
plt.show()

plt.figure()
sns.countplot(train_data.Outlet_Location_Type)
plt.title("Outlet_Location_Type")
plt.show()

plt.figure()
sns.countplot(train_data.Outlet_Type)
plt.xticks(rotation = 90)
plt.title("Outlet_Type")
plt.show()

# Bivariate Analysis: Comparison of features with the target variable
# 1. Item_Weight and Item_Outlet_Sales Analysis: we found out that the correlation was low with the help of a heatmap
plt.figure(figsize=(12,7))
plt.plot(train_data.Item_Weight, train_data["Item_Outlet_Sales"], '.', alpha = 0.3)
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")

# 2. Item_Visibility and Item_Outlet_Sales Analysis: Initial guess based on intuition is that the products that are kept in front will make the sales go high and increase the profit
# Many Products have Visibility = 0
plt.figure(figsize=(12,7))
plt.plot(train_data.Item_Visibility, train_data["Item_Outlet_Sales"], '.', alpha = 0.3)
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Visibility and Item_Outlet_Sales Analysis")
# the data shows a trend that eliminates our hypothesis, which can be due to the fact that important products that control the profit do not need substantial visibility, they are just in demand

# 3. Outlet_Establishment_Year and Item_Outlet_Sales 
Outlet_Establishment_Year_pivot = train_data.pivot_table(index = 'Outlet_Establishment_Year', values = 'Item_Outlet_Sales', aggfunc = np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue', figsize=(12,7))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Item_Outlet_Sales")
plt.title("Outlet_Establishment_Year and Item_Outlet_Sales Analysis")
# Year is not related to the target, year 1998 has low sales which maybe due to the fact that few stores may have opened in that year

# 4. Impact of Item_Fat_Content on Item_Outlet_Sales
Item_Fat_Content_pivot = train_data.pivot_table(index = 'Item_Fat_Content', values = 'Item_Outlet_Sales', aggfunc = np.median)
Item_Fat_Content_pivot.plot(kind='bar', color='blue', figsize=(12,7))
plt.xlabel("Item_Fat_Content")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation = 0)
# low fat products seem to have higher sales than regular products

# 5. Impact of Outlet_Identifier on Item_Outlet_Sales
Outlet_Identifier_pivot = train_data.pivot_table(index = 'Outlet_Identifier', values = 'Item_Outlet_Sales', aggfunc = np.median)
Outlet_Identifier_pivot.plot(kind='bar', color='blue', figsize=(12,7))
plt.xlabel("Outlet_Identifier")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Identifier on Item_Outlet_Sales")
plt.xticks(rotation = 0)
# provided a table to know what each identifier corresponds to
Outlet_Identifier_pivot_info = train_data.pivot_table(values = 'Outlet_Type', columns = 'Outlet_Identifier', aggfunc = lambda x: x.mode())
print(Outlet_Identifier_pivot_info)
# it is visible that Grocery Stores have less sales, maybe because why will someone go for grocery store and then to a different store when there are big stores having everything available under one roof
# Supermarket type 3 has higher sales than Supermarket type 1
Outlet_Type_info = train_data.pivot_table(values = 'Outlet_Type', columns = 'Outlet_Size', aggfunc = lambda x: x.mode())
print(Outlet_Type_info)

# 6. Impact of Outlet_Size on Item_Outlet_Sales
Outlet_Size_pivot = train_data.pivot_table(index = 'Outlet_Size', values = 'Item_Outlet_Sales', aggfunc = np.median)
Outlet_Size_pivot.plot(kind='bar', color='blue', figsize=(12,7))
plt.xlabel("Outlet_Size")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Size on Item_Outlet_Sales")
plt.xticks(rotation = 0)

# 7. Impact of Outlet_Type on Item_Outlet_Sales
Outlet_Type_pivot = train_data.pivot_table(index = 'Outlet_Type', values = 'Item_Outlet_Sales', aggfunc = np.median)
Outlet_Type_pivot.plot(kind='bar', color='blue', figsize=(12,7))
plt.xlabel("Outlet_Type")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Type on Item_Outlet_Sales")
plt.xticks(rotation = 0)

# 8. Impact of Outlet_Location_Type on Item_Outlet_Sales
Outlet_Location_Type_pivot = train_data.pivot_table(index = 'Outlet_Location_Type', values = 'Item_Outlet_Sales', aggfunc = np.median)
Outlet_Location_Type_pivot.plot(kind='bar', color='blue', figsize=(12,7))
plt.xlabel("Outlet_Location_Type")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Location_Type on Item_Outlet_Sales")
plt.xticks(rotation = 0)
# provided a table to know what each identifier corresponds to
Outlet_Location_Type_pivot_info = train_data.pivot_table(values = 'Outlet_Location_Type', columns = 'Outlet_Type', aggfunc = lambda x: x.mode())
print(Outlet_Location_Type_pivot_info)


####################### DATA PREPROCESSING ############################
# Looking for missing values
# first combining the training and testing data sets into one, for data cleaning and feature engineering
# Create 'source' column to later separate the data easily
train_data['source'] = 'train'
test_data['source'] =  'test'
data = pd.concat([train_data, test_data], ignore_index = True)
print(train_data.shape, test_data.shape, data.shape)

# check the number of null values per feature in the combined dataset as a percentage
info_null = (data.isnull().sum() / data.shape[0]) * 100
print(info_null)

# Item_Outlet_Sales is absent in test data set, so no imputing required
# imputing missing values in Item_Weight and Outlet_Size
# we will adopt the mean strategy for imputing the missing data in Item_Weight feature
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer = imputer.fit(data.values[:, 1].reshape(-1, 1))
data.iloc[:, 1] = imputer.transform(data.values[:, 1].reshape(-1, 1))

# we will adopt he mode strategy for imputing the missing data in Outlet_Size feature
imputer1 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

imputer1 = imputer1.fit(data.values[:, 8].reshape(-1, 1))
data.iloc[:, 8] = imputer1.transform(data.values[:, 8].reshape(-1, 1))

info_null_final = (data.isnull().sum() / data.shape[0]) * 100
print(info_null_final)

# this completes DATA PREPROCESSING

############################# FEATURE ENGINEERING ###############################
# fixing the nuances in the data
# considering 0 value in Item_Visibility feature as missing value and imputing the feature
imputer2 = SimpleImputer(missing_values = 0, strategy = 'mean')

imputer2 = imputer2.fit(data.values[:, 3].reshape(-1, 1))
data.iloc[:, 3] = imputer2.transform(data.values[:, 3].reshape(-1, 1))

print(sum(data['Item_Visibility'] == 0))

# Outlet_Establishment_Year has to be modified
# dataset is from 2013
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']

# Item_Type feature has 16 different categories which must be reduced to broader categories for better encoding
# Get the first two characters from the Item_Identifier feature 
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2]) # apply and lambda are used with Pandas dataframe for building complex logic for a new column or filter

# Renaming them to more intuitive broader categories 
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'}) 
info_Item_Type = data['Item_Type_Combined'].value_counts()
print(info_Item_Type)

# Removing the redundancies in the Item_Fat_Content feature
orig_fat_content = data['Item_Fat_Content'].value_counts()
print(orig_fat_content)
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
new_fat_content = data['Item_Fat_Content'].value_counts()
print(new_fat_content)

# making a seperate category 'Non-Edible' for 'Non-Consumable' products in the Item_Fat_Content feature
data.loc[data['Item_Type_Combined'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
new_fat_content = data['Item_Fat_Content'].value_counts()
print(new_fat_content)
print(data.head(n = 10))

############################## Feature Transformation ########################
# convert all categories of categorical variables into numeric tyoes
# we use LabelEncoder() and OneHotEncoder() through ColumnTransformer()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder()

# creating new variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier']) 
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    data[i] = le.fit_transform(data[i])

# Column Transformer for OneHotCoding should not be used as it renders the labeled dataframe into an unlabeled array
# instead use get_dummies function of Pandas, which returns a dummy-coded Dataframe
data = pd.get_dummies(data, columns = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])

# final information about all the features
info_data = data.dtypes
print(info_data)

######################### Exporting Data ###########################
# converting data back into training and test data sets
# dropping the unnecessary columns
data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis = 1, inplace = True)

# divide into train and test data sets
train = data.loc[data['source'] == 'train']
test = data.loc[data['source'] == 'test']

# dropping unnecassary columns
test.drop(['Item_Outlet_Sales','source'], axis=1, inplace=True)
train.drop(['source'], axis=1, inplace=True)

# writing the final info into the text file
buffer = io.StringIO()
train.info(buf=buffer)
s = buffer.getvalue()
with open("train_info.txt", "a", encoding="utf-8") as f:  
    f.write(s)

print(train.head(n = 10))
print(test.head(n = 10))
    
# Export files as modified versions:
train.to_csv("train_modified.csv", index=False)
test.to_csv("test_modified.csv", index=False)
