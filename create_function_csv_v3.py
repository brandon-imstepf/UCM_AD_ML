import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pyDOE import lhs

# Booleans for controlling functionality
testtrain = False  # Control whether to split into train/test
noise = True       # Control whether to add noise
out_of_sample = True  # Generate out-of-sample data
small_sample = False  # Generate smaller sample size

# Booleans for sampling methods
use_lhs = True
use_uniform = True
use_randsample = True

# Parameters for sampling
a = 0
b = 1
c = 20  # number of samples, can be adjusted
abc = f"({a},{b},{c})"
      

# Base directory
basedir = 'C:/Users/brand/Documents/Python Scripts/pysr_code'

# Create a descriptive directory name based on boolean settings
options = []
if testtrain: options.append('tt')
if noise: options.append('n')
if out_of_sample: options.append('oos')
if small_sample: options.append('ss')
options.append(abc)

directory_name = f"RBFs-{'-'.join(options)}"
directory = os.path.join(basedir, directory_name)
os.makedirs(directory, exist_ok=True)

# Import functions from the module
from regression_benchmark_functions import Nguyen_funcs

# Function to generate inputs using LHS
def generate_lhs_inputs(num_vars, num_samples=c):
    samples = lhs(num_vars, samples=num_samples)
    return samples * (b - a) + a  # Rescale to range [a, b]

# Function to generate evenly spaced inputs
def generate_uniform_inputs(num_vars, num_samples=c):
    return np.linspace(a, b, num_samples * num_vars).reshape(num_samples, num_vars)

# Function to generate random sampled inputs
def generate_randsample_inputs(num_vars, num_samples=c):
    return np.random.uniform(a, b, size=(num_samples, num_vars))

# Function to process and save data for each function
def process_functions(functions, prefix, directory, sampling_method, method_name):
    log_lines = []

    for i, func in enumerate(functions, start=1):
        num_vars = func.__code__.co_argcount
        num_samples = 100 if small_sample else c

        # Choose the sampling method
        if sampling_method == 'lhs':
            x = generate_lhs_inputs(num_vars, num_samples)
        elif sampling_method == 'uniform':
            x = generate_uniform_inputs(num_vars, num_samples)
        elif sampling_method == 'randsample':
            x = generate_randsample_inputs(num_vars, num_samples)

        y = np.array([func(*val) for val in x]).reshape(-1, 1)

        if noise:
            noise_std = np.random.uniform(0, 0.1)  # Noise level between 0 and 0.1
            y += np.random.normal(0, noise_std, y.shape)
            log_lines.append(f"{prefix}_F{i}_{method_name}, train/test split, noise_std = {noise_std}")

        columns = [f'x{j+1}' for j in range(num_vars)] + ['y']
        df = pd.DataFrame(np.hstack((x, y)), columns=columns)

        if testtrain:
            # Split the data into training and testing sets
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_filename = f"{prefix}_F{i}_{method_name}_train.csv"
            test_filename = f"{prefix}_F{i}_{method_name}_test.csv"
            train_df.to_csv(os.path.join(directory, train_filename), index=False)
            test_df.to_csv(os.path.join(directory, test_filename), index=False)
        else:
            # Save the entire dataset to a single CSV file
            filename = f"{prefix}_F{i}_{method_name}.csv"
            df.to_csv(os.path.join(directory, filename), index=False)

        # Generate out-of-sample data if required
        if out_of_sample:
            if sampling_method == 'lhs':
                x_oos = generate_lhs_inputs(num_vars, num_samples)
            elif sampling_method == 'uniform':
                x_oos = generate_uniform_inputs(num_vars, num_samples)
            elif sampling_method == 'randsample':
                x_oos = generate_randsample_inputs(num_vars, num_samples)

            y_oos = np.array([func(*val) for val in x_oos]).reshape(-1, 1)
            if noise:
                y_oos += np.random.normal(0, noise_std, y_oos.shape)
            df_oos = pd.DataFrame(np.hstack((x_oos, y_oos)), columns=columns)
            oos_filename = f"{prefix}_F{i}_{method_name}_oos.csv"
            df_oos.to_csv(os.path.join(directory, oos_filename), index=False)

    return log_lines

# List to accumulate log information
log_data = []

# Process the functions for each sampling method if the corresponding boolean is True
if use_lhs:
    log_data += process_functions(Nguyen_funcs, "Nguyen", directory, 'lhs', 'LHS')
if use_uniform:
    log_data += process_functions(Nguyen_funcs, "Nguyen", directory, 'uniform', 'Uniform')
if use_randsample:
    log_data += process_functions(Nguyen_funcs, "Nguyen", directory, 'randsample', 'RandSample')

# Write boolean settings and generated datasets information to the log file
log_file_path = os.path.join(directory, "dataset_info.txt")
with open(log_file_path, "w") as log_file:
    log_file.write(f"testtrain = {testtrain}\n")
    log_file.write(f"noise = {noise}\n")
    log_file.write(f"out_of_sample = {out_of_sample}\n")
    log_file.write(f"small_sample = {small_sample}\n")
    log_file.write(f"use_lhs = {use_lhs}\n")
    log_file.write(f"use_uniform = {use_uniform}\n")
    log_file.write(f"use_randsample = {use_randsample}\n")
    log_file.write("\nDatasets generated:\n")
    log_file.write("\n".join(log_data))
