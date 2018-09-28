import numpy as np
from random import shuffle
from resampling import k_fold_split

def mse(target, predictions):
    return np.mean(np.sum(np.square(target - predictions)))

def r_squared(target, predictions):
    mean_target = np.mean(target)
    return 1 - (np.sum(np.square(target - predictions)))/(np.sum(np.square(target - mean_target)))

def cross_val_mse(regressor, X, target, folds = 5):
    list_of_folds = k_fold_split(X, target, folds)
    computed_values = np.zeros(len(list_of_folds))
    for j, fold in enumerate(list_of_folds):
        regressor.fit(fold['train_X'], fold['train_y'])
        predictions = regressor.predict(fold['test_X'])
        new_mse = mse(fold['test_y'], predictions)
        computed_values[j] = new_mse
    return np.mean(computed_values)
