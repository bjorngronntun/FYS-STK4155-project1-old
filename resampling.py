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

def k_fold_split(X, y, folds = 5):
    data_points = X.shape[0]
    list_of_folds = []
    fold_size = data_points // folds
    indices = list(range(data_points))
    shuffle(indices)

    for k in range(folds - 1):
        fold_indices = indices[k*fold_size : (k + 1)*fold_size]
        non_fold_indices = indices[0: k*fold_size] + indices[(k + 1)*fold_size:]

        list_of_folds.append({
            'test_X': X[fold_indices, :],
            'test_y': y[fold_indices],
            'train_X': X[non_fold_indices, :],
            'train_y': y[non_fold_indices]
        })
        # list_of_folds.append((X[fold_indices, :], y[fold_indices]))
    last_fold_indices = indices[(folds - 1)*fold_size : ]
    
    last_non_fold_indices = indices[: folds*fold_size]
    list_of_folds.append({
        'test_X': X[last_fold_indices, :],
        'test_y': y[last_fold_indices],
        'train_X': X[last_non_fold_indices, :],
        'train_y': y[last_non_fold_indices]
    })
    return list_of_folds
