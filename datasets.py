import numpy as np

# The Franke function:
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def generate_test_data(data_points):
    x = np.random.uniform(high=0, low=1, size=data_points)
    y = np.random.uniform(high=0, low=1, size=data_points)
    # Noise?
    z = FrankeFunction(x, y)
    return np.c_[x, y], z

'''
data, target = generate_test_data(10)
print('Data:', data)
print('Target:', target)
'''