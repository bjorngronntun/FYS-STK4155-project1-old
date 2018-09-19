import numpy as np
from datasets import generate_test_data
from preprocessing import polynomial_combinations
from models import ols_fit, ridge
from assessment import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(X, target)
print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
print('y_train shape', y_train.shape)
print('y_test shape', y_test.shape)
