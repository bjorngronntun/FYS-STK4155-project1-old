import numpy as np
from random import shuffle
def train_test_split(X, y, test_size=0.2):
    data_points = X.shape[0]
    train_points = int(data_points*(1 - test_size))
    indices = list(range(data_points))
    shuffle(indices)
    train_indices = indices[:train_points]
    test_indices = indices[train_points:]
    '''
    print(train_indices)
    print(test_indices)
    print(len(train_indices))
    print(len(test_indices))
    '''
    return X[train_indices, :], X[test_indices, :], y[train_indices], y[test_indices]

def mse(target, predictions):
    return np.mean(np.sum(np.square(target - predictions)))

def r_squared(target, predictions):
    mean_target = np.mean(target)
    return 1 - (np.sum(np.square(target - predictions)))/(np.sum(np.square(target - mean_target)))
