import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pyDOE import lhs

# Parameters and configuration
testtrain = False  # Enable train/test splitting if True
noise = True  # Add noise to the dataset if True
noise_set = True  # Apply noise to specific datasets if True
out_of_sample = True  # Generate out-of-sample data if True
small_sample = False  # Use a small sample size if True

use_lhs = True  # Use Latin Hypercube Sampling
use_uniform = True  # Use uniform sampling
use_randsample = True  # Use random sampling

a, b, c = 3, 5, 20  # Domain range [a, b] and sample size c
abc = f"({a},{b},{c})"

noise_levels = [0, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5]  # Different levels of noise
basedir = 'C:/Users/brand/Documents/Python Scripts/pysr_code'
options = []

# Directory naming conventions
func_name = "Imstepf"  # Name for function set
if testtrain: options.append('tt')
if noise: options.append('n')
if noise_set: options.append('ns')
if out_of_sample: options.append('oos')
if small_sample: options.append('ss')
options.append(abc)

directory_name = f"{func_name}-RBFs-{'-'.join(options)}"
directory = os.path.join(basedir, directory_name)
os.makedirs(directory, exist_ok=True)

# Import functions from the module
from regression_benchmark_functions import Imstepf_funcs

# Helper functions
def generate_lhs_inputs(num_vars, num_samples):
    """Generate LHS inputs within the domain [a, b]."""
    return np.round( 
        lhs(num_vars, samples=num_samples) * (b - a) + a
        ,4)

def generate_uniform_inputs(num_vars, num_samples):
    """Generate uniformly spaced inputs."""
    return np.round(
        np.linspace(a, b, num_samples * num_vars).reshape(num_samples, num_vars)
        ,4)

def generate_randsample_inputs(num_vars, num_samples):
    """Generate random inputs within [a, b]."""
    return np.round(
        np.random.uniform(a, b, size=(num_samples, num_vars))
        ,4)

def generate_oos_range():
    """Generate a new range for out-of-sample data, offset from [a, b]."""
    c_new = np.round(
        np.random.uniform(a + b, 100)
        ,4)
    d_new = c_new + (b - a)
    return c_new, d_new

# Detailed logging function
def generate_log_info(log_file_path, log_data, noise_levels, sampling_methods):
    """Create and save a detailed log file with dataset configurations."""
    with open(log_file_path, "w") as log_file:
        log_file.write(f"testtrain = {testtrain}\n")
        log_file.write(f"noise = {noise}\n")
        log_file.write(f"out_of_sample = {out_of_sample}\n")
        log_file.write(f"small_sample = {small_sample}\n")
        log_file.write(f"use_lhs = {use_lhs}\n")
        log_file.write(f"use_uniform = {use_uniform}\n")
        log_file.write(f"use_randsample = {use_randsample}\n")
        log_file.write(f"Noise Levels: {noise_levels}\n\n")
        log_file.write("Datasets generated:\n")
        log_file.write("\n".join(log_data))

# Function to process datasets with noise and sampling details
def process_functions(functions, prefix, directory, sampling_method, method_name, noise_levels):
    log_lines = []

    for noise_std in noise_levels:
        noise_dir = os.path.join(directory, f"{int(noise_std * 100)}_Percent_Noise")
        os.makedirs(noise_dir, exist_ok=True)

        for i, func in enumerate(functions, start=1):
            num_vars = func.__code__.co_argcount
            num_samples = 100 if small_sample else c
            x = y = None
            data_range = (a, b)  # default data range

            # Choose sampling method and generate inputs
            if sampling_method == 'lhs':
                x = generate_lhs_inputs(num_vars, num_samples)
            elif sampling_method == 'uniform':
                x = generate_uniform_inputs(num_vars, num_samples)
            elif sampling_method == 'randsample':
                x = generate_randsample_inputs(num_vars, num_samples)

            # Calculate function outputs with noise if required
            y = np.round(
                np.array([func(*val) for val in x]).reshape(-1, 1)
                ,4)
            if noise:
                y += np.random.normal(0, noise_std, y.shape)
            log_lines.append(f"{prefix}_F{i}_{method_name} with noise_std = {noise_std}, range = {data_range}")

            # Save dataset as CSV
            columns = [f'x{j+1}' for j in range(num_vars)] + ['y']
            df = pd.DataFrame(np.hstack((x, y)), columns=columns)
            filename = f"{prefix}_F{i}_{method_name}.csv"
            df.to_csv(os.path.join(noise_dir, filename), index=False)

            # Generate out-of-sample data if required
            if out_of_sample:
                c_new, d_new = generate_oos_range()
                data_range = (c_new, d_new)
                x_oos = np.random.uniform(c_new, d_new, size=(num_samples, num_vars))
                y_oos = np.array([func(*val) for val in x_oos]).reshape(-1, 1)
                if noise:
                    y_oos += np.random.normal(0, noise_std, y_oos.shape)
                df_oos = pd.DataFrame(np.hstack((x_oos, y_oos)), columns=columns)
                oos_filename = f"{prefix}_F{i}_{method_name}_oos.csv"
                df_oos.to_csv(os.path.join(noise_dir, oos_filename), index=False)
                log_lines.append(f"{prefix}_F{i}_{method_name}_oos with noise_std = {noise_std}, range = {data_range}")

    return log_lines

# List to accumulate log information
log_data = []

# Process the functions for each sampling method
sampling_methods = [('lhs', 'LHS'), ('uniform', 'Uniform'), ('randsample', 'RandSample')]
for sampling_key, method_name in sampling_methods:
    if (sampling_key == 'lhs' and use_lhs) or \
       (sampling_key == 'uniform' and use_uniform) or \
       (sampling_key == 'randsample' and use_randsample):
        log_data += process_functions(Imstepf_funcs, func_name, directory, sampling_key, method_name, noise_levels)

# Write detailed log information
log_file_path = os.path.join(directory, "dataset_info.txt")
generate_log_info(log_file_path, log_data, noise_levels, sampling_methods)
