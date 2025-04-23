import numpy as np
from sklearn import metrics

# Error calculation functions with NaN handling
def calculate_mse_nmse(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.nan, np.nan

    mse = metrics.mean_squared_error(y_true, y_pred)
    var_y_true = np.var(y_true)

    if var_y_true == 0:
        nmse = np.nan  # Avoid division by zero
    else:
        nmse = mse / var_y_true

    return mse, nmse

def calculate_mse(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.nan
    return metrics.mean_squared_error(y_true, y_pred)

def calculate_nmse(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.nan

    mse = metrics.mean_squared_error(y_true, y_pred)
    var_y_true = np.var(y_true)

    if var_y_true == 0:
        return np.nan  # Avoid division by zero
    return mse / var_y_true

def calculate_rmse(y_true, y_pred):
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        return np.nan
    mse = metrics.mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)
