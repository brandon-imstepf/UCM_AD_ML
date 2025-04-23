import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import os
import pandas as pd

base_directory = r'C:\Users\brand\Desktop\Raj-Sindi\training_data\sim_csv_v10_biased'

os.chdir(base_directory)
csv_name = 'data_frac_07_widerange_biased_gamma_k2.csv'
data = pd.read_csv(csv_name)

# Generate histogram of data with rows of 3 subplots
num_columns = len(data.columns)
rows = (num_columns + 2) // 3  # Calculate the number of rows needed
fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

# Flatten axes for easier indexing, handle cases where axes is not a 2D array
axes = axes.flatten() if rows > 1 else np.array([axes]).flatten()

for i, column in enumerate(data.columns):
    ax = axes[i]
    ax.hist(data[column], bins=30, alpha=0.7)
    ax.set_title(column)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# Hide any unused subplots
for j in range(len(data.columns), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

