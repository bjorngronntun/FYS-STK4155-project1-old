import numpy as np
from datasets import generate_test_data
from preprocessing import polynomial_combinations
from models import ols_fit, ridge

data_points = 100
max_degree = 5

initial_data, target = generate_test_data(50, 0.1)
X = polynomial_combinations(initial_data, max_degree)

beta_hat_ols = ols_fit(X, target)
predictions_ols = np.dot(X, beta_hat_ols)
beta_hat_ridge = ridge(X, target, 0.01)
predictions_ridge = np.dot(X, beta_hat_ridge)

# This actually looks good...
MSE_ols = np.mean(np.sum(np.square(target - predictions_ols)))
MSE_ridge = np.mean(np.sum(np.square(target - predictions_ridge)))
print('MSE OLS', MSE_ols)
print('MSE Ridge', MSE_ridge)
