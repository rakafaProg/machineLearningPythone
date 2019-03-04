# Data Preprocessing Template

# math tools
import numpy as np
# to plot charts
import matplotlib.pyplot as plt
# import and manage dataset
import pandas as pd

# ===============================
#     Importing the data set
# ===============================

fileName = 'Preprocessing_Data'
featuresLength = 3
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
#    Take care of missing Data
# ===============================

# mark in range of columns where you have missing data
rangeStart = 1
rangeEnd = 2

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# 1:3 takes 1-2
imputer = imputer.fit(X[:, rangeStart:rangeEnd+1])
X[:, rangeStart:rangeEnd+1] = imputer.transform(X[:, rangeStart:rangeEnd+1])

# Clear range varialbes
del rangeStart
del rangeEnd


# ===============================
#    Encode categorical data
# ===============================

# Choose which column needs encoding
columnNum = 0

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

# Encoding the Dependent Variable - 
# In case of 'Yes' 'No' values, no need to dommy encoding
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# clear variables 
del columnNum



# ===============================
#    Split to train and test
# ===============================

testSize = 0.2

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)

del testSize


# ===============================
#        Feature Scaling
# ===============================

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Fit only on train set 
# so train and test will be scaled on the same basis 
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_test = sc_y.transform(y_test.reshape(-1,1))

"""
# Later -> predict specific result, with reverse scaling
X_test = [[6.6]]
X_test = sc_X.transform(X_test)

y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
"""


