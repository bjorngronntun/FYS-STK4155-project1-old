import numpy as np
from random import shuffle
from resampling import k_fold_split
'''
def variance_of_sigma(target, predictions, n, p):
    return np.sum(np.square(target - predictions))/(n - p - 1)

def variance_of_beta(X, target, beta):
    predictions = np.dot(X, beta)
'''
def mse(target, predictions):
    return np.mean(np.sum(np.square(target - predictions)))

def r_squared(target, predictions):
    mean_target = np.mean(target)
    return 1 - (np.sum(np.square(target - predictions)))/(np.sum(np.square(target - mean_target)))

def cross_val_mse_and_r_squared(regressor, X, target, folds = 5):
    list_of_folds = k_fold_split(X, target, folds)
    computed_mses = np.zeros(len(list_of_folds))
    computed_r_squareds = np.zeros(len(list_of_folds))
    for j, fold in enumerate(list_of_folds):
        regressor.fit(fold['train_X'], fold['train_y'])
        predictions = regressor.predict(fold['test_X'])
        new_mse = mse(fold['test_y'], predictions)
        new_r_squared = r_squared(fold['test_y'], predictions)
        computed_mses[j] = new_mse
        computed_r_squareds[j] = new_r_squared
    return np.mean(computed_mses), np.mean(computed_r_squareds)
