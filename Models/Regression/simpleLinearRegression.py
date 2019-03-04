# Simple Linear Regression

# math tools
import numpy as np
# to plot charts
import matplotlib.pyplot as plt
# import and manage dataset
import pandas as pd

# ===============================
#     Importing the data set
# ===============================

fileName = 'Salary_Data'
featuresLength = 1
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
#    Split to train and test
# ===============================

testSize = 1/3

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

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Test via Prediction
plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_test, y_pred, color = 'green')
plt.title('Test via Prediction (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Compare results to test
compareRes = np.column_stack((y_test, y_pred))


