# Fama french model project skeleton code

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load your portfolio returns and factor data
portfolio_returns = pd.read_csv('portfolio_returns.csv', index_col='Date', parse_dates=True)
factors = pd.read_csv('factor_data.csv', index_col='Date', parse_dates=True)

# Align dates and calculate excess returns
rf = factors['RF']
excess_portfolio_returns = portfolio_returns['Portfolio'] - rf

# Prepare the independent variables (market, SMB, HML for Fama-French 3-Factor)
X = factors[['Mkt-RF', 'SMB', 'HML']]
X = sm.add_constant(X)  # Add alpha (intercept) term

# Fit the factor model (regression)
model = sm.OLS(excess_portfolio_returns, X).fit()

# Print summary of the regression (factor loadings, alpha, etc.)
print(model.summary())

# Extract betas and residuals
betas = model.params
residuals = model.resid

# Variance attribution (factor risk + residual risk)
factor_var = np.var(np.dot(X, betas), axis=0)
residual_var = np.var(residuals)
total_var = factor_var + residual_var

print("Factor variance:", factor_var)
print("Residual variance:", residual_var)
print("Total variance:", total_var)

# You can also plot the betas and attribution here using matplotlib
