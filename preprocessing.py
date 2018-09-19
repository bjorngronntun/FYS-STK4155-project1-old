import numpy as np
def polynomial_combinations(input_data, max_degree):
    data_points = input_data.shape[0]
    # This is bad, as it really works only with two-dimensional
    # input data:
    x = input_data[:, 0]
    y = input_data[:, 1]
    X = np.zeros((data_points, 0))
    for i in range(max_degree + 1):
        for j in range(max_degree + 1):
            X = np.c_[X, (x**i)*(y**j)]
    return X
