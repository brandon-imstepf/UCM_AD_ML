import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
import matplotlib.pyplot as plt
import os

# Define your scaling function with return of fitted transformers
def scale_data_with_transformers(x, y):
    # Apply Yeo-Johnson power transformation (works with both positive and negative data)
    power_transformer_x = PowerTransformer(method='yeo-johnson')
    power_transformer_y = PowerTransformer(method='yeo-johnson')

    x_transformed = power_transformer_x.fit_transform(x)
    y_transformed = power_transformer_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Scale the data to a standard range (e.g., zero mean, unit variance)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_scaled = scaler_x.fit_transform(x_transformed)
    y_scaled = scaler_y.fit_transform(y_transformed.reshape(-1, 1)).flatten()

    return x_scaled, y_scaled, power_transformer_x, power_transformer_y, scaler_x, scaler_y

# Load the ground truth and estimation CSVs (example paths)
base_dir = "C:/Users/brand/Desktop/Raj-Sindi/Tau_Transport-jtorok_dev/SampleFiles/11-15"
os.chdir(base_dir)
ground_truth_df = pd.read_csv('11-15_test.csv')
estimation_df = pd.read_csv('11-15_test_sr.csv')

# Ensure the FValue column exists in both DataFrames
if 'FValue' not in ground_truth_df.columns or 'FValue' not in estimation_df.columns:
    raise ValueError("Missing 'FValue' column in one of the input CSVs.")

# Separate FValue from the other features
y_true = ground_truth_df['FValue'].values  # Ground truth flux
y_pred = estimation_df['FValue'].values  # Predicted flux (estimation)

# Apply scaling to flux values and retrieve transformers
_, y_true_scaled, _, power_transformer_y_true, _, scaler_y_true = scale_data_with_transformers(
    ground_truth_df.drop(columns=['FValue']), y_true
)
_, y_pred_scaled, _, power_transformer_y_pred, _, scaler_y_pred = scale_data_with_transformers(
    estimation_df.drop(columns=['FValue']), y_pred
)

# Calculate residuals (scaled values for consistency)
residuals_scaled = y_true_scaled - y_pred_scaled

# De-transform flux values to their original scale
y_true_detransformed = scaler_y_true.inverse_transform(
    power_transformer_y_true.inverse_transform(y_true_scaled.reshape(-1, 1))
).flatten()
y_pred_detransformed = scaler_y_pred.inverse_transform(
    power_transformer_y_pred.inverse_transform(y_pred_scaled.reshape(-1, 1))
).flatten()

# Calculate residuals for de-transformed data
residuals_detransformed = y_true_detransformed - y_pred_detransformed

# Plotting residuals (scaled)
plt.figure(figsize=(10, 6))
plt.scatter(y_true_scaled, residuals_scaled, color='blue', alpha=0.5, label='Residuals (Scaled)')
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
plt.xlabel('Scaled Ground Truth FValue')
plt.ylabel('Residuals (Ground Truth - Estimation)')
plt.title('Residuals (Scaled) between Ground Truth and Estimation')
plt.legend()
plt.grid(True)
plt.show()

# Plotting residuals (de-transformed)
plt.figure(figsize=(10, 6))
plt.scatter(y_true_detransformed, residuals_detransformed, color='purple', alpha=0.5, label='Residuals (Original Scale)')
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
plt.xlabel('Original Ground Truth FValue')
plt.ylabel('Residuals (Ground Truth - Estimation)')
plt.title('Residuals (Original Scale) between Ground Truth and Estimation')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Ground Truth vs. Estimation (scaled)
plt.figure(figsize=(10, 6))
plt.scatter(y_true_scaled, y_pred_scaled, color='green', alpha=0.5, label='Predictions (Scaled)')
plt.plot(
    [min(y_true_scaled), max(y_true_scaled)],
    [min(y_true_scaled), max(y_true_scaled)],
    color='black',
    linestyle='--',
    label='Perfect Prediction (y=x)'
)
plt.xlabel('Scaled Ground Truth FValue')
plt.ylabel('Scaled Estimated FValue')
plt.title('Ground Truth vs. Estimation (Scaled)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Ground Truth vs. Estimation (de-transformed)
plt.figure(figsize=(10, 6))
plt.scatter(y_true_detransformed, y_pred_detransformed, color='orange', alpha=0.5, label='Predictions (Original Scale)')
plt.plot(
    [min(y_true_detransformed), max(y_true_detransformed)],
    [min(y_true_detransformed), max(y_true_detransformed)],
    color='black',
    linestyle='--',
    label='Perfect Prediction (y=x)'
)
plt.xlabel('Original Ground Truth FValue')
plt.ylabel('Original Estimated FValue')
plt.title('Ground Truth vs. Estimation (Original Scale)')
plt.legend()
plt.grid(True)
plt.show()
