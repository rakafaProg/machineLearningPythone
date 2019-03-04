
# Random Forest Regression

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
#        Fit the model
# ===============================

# Fitting the model to the dataset
from sklearn.ensemble import RandomForestRegressor
# n_estimators -> How many trees will be created.
# The more trees, the more accurate the result
regressor = RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(X,y)

# Predicting a new result
X_test = [[6.6]]
y_pred = regressor.predict(X_test)

# ===============================
#      Visualise the results
# ===============================

# Visualising the Decision Tree results
"""plt.scatter(X, y, color='green')
plt.plot(X, regressor.predict(X), 'gray')
plt.show()"""

# Visualising the Decision Tree results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'grey')
plt.show()

