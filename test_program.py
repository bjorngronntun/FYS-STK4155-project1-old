import numpy as np
from datasets import generate_test_data
from preprocessing import polynomial_combinations
from models import OLS, Ridge
from resampling import k_fold_split
from assessment import cross_val_mse

data_points = 100
max_degree = 5

initial_data, target = generate_test_data(50, 0.1)
X = polynomial_combinations(initial_data, max_degree)

print(X)
ols_regressor = OLS()
ridge_regressor = Ridge(0.01)

ols_regressor.fit(X, target)
ridge_regressor.fit(X, target)

folds = 7
list_of_folds = k_fold_split(X, target, folds)

print(len(list_of_folds))
print(cross_val_mse(ridge_regressor, X, target, folds))
