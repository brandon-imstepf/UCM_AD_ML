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

# Randomly permute the rows before any further processing
master_data = master_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Optionally trim down to every nth point (e.g., every 100th row) for validation set
if trim:
    validation_data = master_data.iloc[::20].reset_index(drop=True)
    # Remove validation data from the master data (keep only non-trimmed rows)
    master_data = master_data.drop(master_data.index[::20]).reset_index(drop=True)

# Split the remaining data into training and testing sets
# You can modify test_size to change the ratio of the split (currently set to 20%)
train_data, test_data = train_test_split(master_data, test_size=0.20, random_state=42)

# Print the number of rows and columns for each set
print(f'Training Data - Rows: {len(train_data)}, Columns: {len(train_data.columns)}')
print(f'Testing Data - Rows: {len(test_data)}, Columns: {len(test_data.columns)}')
print(f'Validation Data - Rows: {len(validation_data)}, Columns: {len(validation_data.columns)}')

# Create the output file names
train_output_file_name = basedir + 'allparams_training.csv'
test_output_file_name = basedir + 'allparams_testing.csv'
validation_output_file_name = basedir + 'allparams_validation.csv'

# Save the train, test, and validation data to new CSV files
train_data.to_csv(train_output_file_name, index=False)
test_data.to_csv(test_output_file_name, index=False)
validation_data.to_csv(validation_output_file_name, index=False)

print(f'Train data saved to {train_output_file_name}')
print(f'Test data saved to {test_output_file_name}')
print(f'Validation data saved to {validation_output_file_name}')
