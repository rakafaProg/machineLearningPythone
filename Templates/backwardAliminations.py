
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