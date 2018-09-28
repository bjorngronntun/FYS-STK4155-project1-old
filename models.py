import numpy as np
"""
class Regressor:
    def __init__(self):
        self.fitted = False
    def fit(self, X, target):
        pass
    def predict(self, X):
        pass
"""
class OLS:
    def __init__(self):
        pass
    def fit(self, X, target):
        self.beta_hat_ = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), target)
    def predict(self, X):
        return np.dot(X, self.beta_hat_)

class Ridge:
    def __init__(self, lam):
        self.lam = lam
    def fit(self, X, target):
        p = X.shape[1]
        self.beta_hat_ = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X) + self.lam*np.eye(p)), X.transpose()), target)
    def predict(self, X):
        return np.dot(X, self.beta_hat_)
