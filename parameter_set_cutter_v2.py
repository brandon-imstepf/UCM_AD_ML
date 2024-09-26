import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Brandon Imstepf

# Define directories and file paths
basedir = 'C:/Users/brand/Documents/MATLAB/Raj_Torok Lab/sim_csv_v5/Sept 16/'

# File name for the master concatenated file
master_file = basedir + 'master_flux_data.csv'

# Boolean to trim every nth row
trim = True

# Load the master concatenated data
master_data = pd.read_csv(master_file)

# Print the number of rows and columns of the master CSV
print(f'Master File: {master_file} - Rows: {len(master_data)}, Columns: {len(master_data.columns)}')

# Optionally trim down to every nth point (e.g., every 100th row)
if trim:
    master_data = master_data.iloc[::20].reset_index(drop=True)

# Use scikit-learn to split the data into training and testing sets
# You can modify test_size to change the ratio of the split (currently set to 20%)
train_data, test_data = train_test_split(master_data, test_size=0.20, random_state=42)

# Print the number of rows and columns for both sets
print(f'Training Data - Rows: {len(train_data)}, Columns: {len(train_data.columns)}')
print(f'Testing Data - Rows: {len(test_data)}, Columns: {len(test_data.columns)}')

# Create the output file names
cut_output_file_name = basedir + 'allparams_testing.scikit' + '.csv'
remaining_output_file_name = basedir + 'allparams_training.scikit' + '.csv'

# Save the test and train data to new CSV files
test_data.to_csv(cut_output_file_name, index=False)
train_data.to_csv(remaining_output_file_name, index=False)

print(f'Test data saved to {cut_output_file_name}')
print(f'Train data saved to {remaining_output_file_name}')
