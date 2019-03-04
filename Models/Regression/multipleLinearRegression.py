# Multi Linear Regression

# math tools
import numpy as np
# to plot charts
import matplotlib.pyplot as plt
# import and manage dataset
import pandas as pd

# ===============================
#     Importing the data set
# ===============================

fileName = '50_Startups'
featuresLength = 4
baseUrl = '/Users/rakefetyifrach/Desktop/python-machine-learning/DataExamples/'

# Importing the dataset
dataset = pd.read_csv(baseUrl + fileName + '.csv')

# Create matrix of the features
X = dataset.iloc[:, :-1].values
# Create vector of values
y = dataset.iloc[:, featuresLength].values


# After importing - clear unused variables
del fileName
del baseUrl
del featuresLength

# ===============================
#    Encode categorical data
# ===============================

# Choose which column needs encoding
columnNum = 3

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Encode the choosen column (country in the example)
X[:, columnNum] = labelencoder_X.fit_transform(X[:, columnNum])
# Dommy encoding - create column for each type of encoded value, and set it as 1 / 0
# makes this coulmn to categoical column
onehotencoder = OneHotEncoder(categorical_features = [columnNum])
X = onehotencoder.fit_transform(X).toarray()
# Avoid dummy variable trap
X = X[:, 1:]


# ===============================
#    Split to train and test
# ===============================

testSize = 0.2

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)

del testSize


# ===============================
#        Fit the model
# ===============================

# Fitting Simple Linear Regression to the Training set
# No need to scale. The LinearRegression class does that automaticaly.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# ===============================
#      Visualise the results
# ===============================

""" Cannot visualise - too many features... """
# Compare results to test
compareRes = np.column_stack((y_test, y_pred))


# ===============================
#   Optimize features choosing
# ===============================
# backward aliminations
import statsmodels.formula.api as sm
X = np.append( arr=np.ones((50,1)).astype(int), values=X, axis=1)

X_opt = X[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()

regressor_ols.summary()

# x2 - 0.99 p-value -> will be removed
X_opt = X[:, [0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()

regressor_ols.summary()

X_opt = X[:, [0,3]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()

regressor_ols.summary()

"""
Each time remove the x that has the bigger p-value, 
and see if the Adj. R-squared Got bigger.
Skew - if negative - meens increasing this x will decrease y, 
    if positive - increasing x will increase y.
    
"""
