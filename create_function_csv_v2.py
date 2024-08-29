import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pyDOE import lhs

# Booleans for controlling functionality
testtrain = False  # Control whether to split into train/test
noise = True      # Control whether to add noise
out_of_sample = True  # Generate out-of-sample data
small_sample = False   # Generate smaller sample size

# Base directory
basedir = 'C:/Users/brand/Documents/Python Scripts/pysr_code'

# Create a descriptive directory name based on boolean settings
options = []
if testtrain: options.append('tt')
if noise: options.append('n')
if out_of_sample: options.append('oos')
if small_sample: options.append('ss')

directory_name = f"RBFs-{'-'.join(options)}"
directory = os.path.join(basedir, directory_name)
os.makedirs(directory, exist_ok=True)

# Import functions from the module
from regression_benchmark_functions import Nguyen_funcs

# Function to generate inputs using LHS
def generate_inputs(num_vars, num_samples=1000):
    samples = lhs(num_vars, samples=num_samples)
    return samples * (100.1 - 0.1) + 0.1  # Rescale to range [0.1, 100.1]

# Function to process and save data for each function
def process_functions(functions, prefix, directory):
    log_lines = []

    for i, func in enumerate(functions, start=1):
        num_vars = func.__code__.co_argcount
        num_samples = 100 if small_sample else 1000
        x = generate_inputs(num_vars, num_samples)
        y = np.array([func(*val) for val in x]).reshape(-1, 1)

        if noise:
            noise_std = np.random.uniform(0, 0.1)  # Noise level between 0 and 0.1
            y += np.random.normal(0, noise_std, y.shape)
            log_lines.append(f"{prefix}_F{i}, train/test split, noise_std = {noise_std}")

        columns = [f'x{j+1}' for j in range(num_vars)] + ['y']
        df = pd.DataFrame(np.hstack((x, y)), columns=columns)

        if testtrain:
            # Split the data into training and testing sets
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_filename = f"{prefix}_F{i}_train.csv"
            test_filename = f"{prefix}_F{i}_test.csv"
            train_df.to_csv(os.path.join(directory, train_filename), index=False)
            test_df.to_csv(os.path.join(directory, test_filename), index=False)
        else:
            # Save the entire dataset to a single CSV file
            filename = f"{prefix}_F{i}.csv"
            df.to_csv(os.path.join(directory, filename), index=False)

        # Generate out-of-sample data if required
        if out_of_sample:
            x_oos = generate_inputs(num_vars, num_samples)
            y_oos = np.array([func(*val) for val in x_oos]).reshape(-1, 1)
            if noise:
                y_oos += np.random.normal(0, noise_std, y_oos.shape)
            df_oos = pd.DataFrame(np.hstack((x_oos, y_oos)), columns=columns)
            oos_filename = f"{prefix}_F{i}_oos.csv"
            df_oos.to_csv(os.path.join(directory, oos_filename), index=False)

    # Write boolean settings and generated datasets information to the log file
    log_file_path = os.path.join(directory, "dataset_info.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"testtrain = {testtrain}\n")
        log_file.write(f"noise = {noise}\n")
        log_file.write(f"out_of_sample = {out_of_sample}\n")
        log_file.write(f"small_sample = {small_sample}\n")
        log_file.write("\nDatasets generated:\n")
        log_file.write("\n".join(log_lines))

# Process the functions and save datasets
process_functions(Nguyen_funcs, "Nguyen", directory)
