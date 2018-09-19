import numpy as np
def ols_fit(X, target):
    beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), target)
    return beta_hat

def ridge(X, target, lam):
    p = X.shape[1]
    print(p)
    beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X) + lam*np.eye(p)), X.transpose()), target)
    return beta_hat
