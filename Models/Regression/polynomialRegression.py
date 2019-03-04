# Polynomial Linear Regression

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

# Create the simple linear regression (for later comparing)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
# Transform X to contain x1^2, x2^3, x3^4 etc. (keeping the original Xs)
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# Create poly model with the transformed X
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# ===============================
#      Visualise the results
# ===============================

# plot linear result
plt.scatter(X, y, color='green')
plt.plot(X, lin_reg.predict(X), 'gray')
plt.title('Linear Regression')
plt.show()

# plot polynomial result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='green')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), 'blue')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), 'red')
plt.title('Polynomial Regression')
plt.show()

# ===============================
#     Predict on single item
# ===============================

# Predict (Truth or Bluff?)
pred = [[6.5]]
lin_reg.predict(pred)
lin_reg2.predict(poly_reg.fit_transform(pred))

