import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
import matplotlib.pyplot as plt
import os
from stat_plots import scale_data

# Load the ground truth and estimation CSVs (example paths)
base_dir = "C:/Users/brand/Desktop/Raj-Sindi/Tau_Transport-jtorok_dev/SampleFiles/11-22"
os.chdir(base_dir)
ground_truth_df = pd.read_csv('11-15_test.csv')
estimation_df = pd.read_csv('11-22_test_sr_UntouchedGlobals.csv')

# Ensure the FValue column exists in both DataFrames
if 'FValue' not in ground_truth_df.columns or 'FValue' not in estimation_df.columns:
    raise ValueError("Missing 'FValue' column in one of the input CSVs.")

# Extract true and predicted values
y_true = ground_truth_df['FValue'].values  # Ground truth flux
y_pred = estimation_df['FValue'].values  # Predicted flux (already transformed)

# Fit the PowerTransformer to the ground truth (y_true)
power_transformer = PowerTransformer(method='yeo-johnson')
y_true_transformed = power_transformer.fit_transform(y_true.reshape(-1, 1)).flatten()

# Transform y_pred using the same transformer
#y_pred_transformed = power_transformer.transform(y_pred.reshape(-1, 1)).flatten()

# De-transform y_pred to the original scale for comparison
y_pred_original_scale = power_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Calculate residuals
ground_residuals = y_true - y_pred_original_scale
transformed_residuals = y_true_transformed - y_pred


# Troubleshooting output
print("Sum of y_true_transformed:", sum(y_true_transformed))
print("Sum of y_pred_original_scale:", sum(y_pred_original_scale))

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_true, ground_residuals, color='blue', alpha=0.5, label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
plt.xlabel('Ground Truth FValue (Original Scale)')
plt.ylabel('Residuals (Ground Truth - Estimation)')
plt.title('Ground Truth Scale Residuals')
plt.legend()
plt.grid(True)
plt.show()

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_true_transformed, transformed_residuals, color='blue', alpha=0.5, label='Residuals')
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
plt.xlabel('Ground Truth FValue (Transformed Scale)')
plt.ylabel('Residuals (Ground Truth - Estimation)')
plt.title('Transformed Scale Residuals')
plt.legend()
plt.grid(True)
plt.show()

# Plot Ground Truth vs. Estimation in original scale
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred_original_scale, color='green', alpha=0.5, label='Predictions')
plt.plot(
    [min(y_true), max(y_true)],
    [min(y_true), max(y_true)],
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
