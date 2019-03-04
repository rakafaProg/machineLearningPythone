import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fileName = 'Position_Salaries.csv'
X_len = 2
testSize = 0.2

# Importing the dataset
dataset = pd.read_csv(fileName)
X = dataset.iloc[:, -1].values
y = dataset.iloc[:, X_len].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

X_test = [[6.5]]

y_pred = regressor.predict(X_test)

plt.scatter(X, y, color='green')
plt.plot(X, regressor.predict(X), 'gray')
plt.show()

