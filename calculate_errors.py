import numpy as np
from sklearn import metrics

# Error calculation functions
def calculate_mse_nmse(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    nmse = mse / np.var(y_true)
    return mse, nmse

def calculate_mse(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    return mse

def calculate_nmse(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    nmse = mse / np.var(y_true)
    return nmse
