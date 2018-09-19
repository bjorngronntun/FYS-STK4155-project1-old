import numpy as np
from datasets import generate_test_data
from preprocessing import polynomial_combinations
from models import ols_fit

data_points = 100
max_degree = 5

initial_data, target = generate_test_data(50)
X = polynomial_combinations(initial_data, max_degree)
beta_hat = ols_fit(X, target)

predictions = np.dot(X, beta_hat)

# This actually looks good...
MSE = np.mean(np.sum(np.square(target - predictions)))
print('MSE', MSE)
