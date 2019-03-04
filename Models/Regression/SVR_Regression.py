
# Support Vector Regression (SVR)

# math tools
import numpy as np
# to plot charts
import matplotlib.pyplot as plt
# import and manage dataset
import pandas as pd

# ===============================
#     Importing the data set
# ===============================

fileName = 'Position_Salaries'
featuresLength = 2
baseUrl = '/Users/rakefetyifrach/Desktop/python-machine-learning/DataExamples/'

# Importing the dataset
dataset = pd.read_csv(baseUrl + fileName + '.csv')

# Create matrix of the features
X = dataset.iloc[:, 1:2].values
# Create vector of values
y = dataset.iloc[:, featuresLength].values


# After importing - clear unused variables
del fileName
del baseUrl
del featuresLength


# ===============================
#    Split to train and test
# ===============================
"""
# The example data is too small - 
# so there is no need for splitting
testSize = 0.2

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)

del testSize
"""

# ===============================
#        Feature Scaling
# ===============================

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
# X_test = sc_X.transform(X_test)
# Fit only on train set 
# so train and test will be scaled on the same basis 
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1,1))
# y_test = sc_y.transform(y_test.reshape(-1,1))

"""
# Later -> predict specific result, with reverse scaling
X_test = [[6.6]]
X_test = sc_X.transform(X_test)

y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
"""


# ===============================
#        Fit the model
# ===============================
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', gamma='auto')
regressor.fit(X, y)


# ===============================
#      Visualise the results
# ===============================

plt.scatter(X, y, color='green')
plt.plot(X, regressor.predict(X), 'gray')
plt.title('Support Vector Regression (SVR)')
plt.show()

# ===============================
#     Predict on single item
# ===============================

# Predict (Truth or Bluff?)
X_test = [[6.5]]
X_test = sc_X.transform(X_test)

y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

print ('Prediction for 6.5 is: ')
print (y_pred[0])