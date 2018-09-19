import numpy as np
from datasets import generate_test_data
from preprocessing import polynomial_combinations
from models import ols_fit, ridge
from assessment import train_test_split, mse, r_squared

data_points = 100
max_degree = 5

initial_data, target = generate_test_data(50, 0.1)
X = polynomial_combinations(initial_data, max_degree)

beta_hat_ols = ols_fit(X, target)
predictions_ols = np.dot(X, beta_hat_ols)
beta_hat_ridge = ridge(X, target, 0.01)
predictions_ridge = np.dot(X, beta_hat_ridge)

# This actually looks good...
MSE_ols = mse(target, predictions_ols)
MSE_ridge = mse(target, predictions_ridge)
print('MSE OLS', MSE_ols)
print('MSE Ridge', MSE_ridge)
print()
print('Now for the great train-test-split')
X_train, X_test, y_train, y_test = train_test_split(X, target)
beta_hat_ols = ols_fit(X_train, y_train)
predictions_ols = np.dot(X_test, beta_hat_ols)
beta_hat_ridge = ridge(X_train, y_train, 0.01)
predictions_ridge = np.dot(X_test, beta_hat_ridge)
MSE_ols = mse(y_test, predictions_ols)
MSE_ridge = mse(y_test, predictions_ridge)
r_squared_ols = r_squared(y_test, predictions_ols)
r_squared_ridge = r_squared(y_test, predictions_ridge)
print('MSE OLS', MSE_ols)
print('MSE Ridge', MSE_ridge)
print('R squared OLS', r_squared_ols)
print('R squared Ridge', r_squared_ridge)
