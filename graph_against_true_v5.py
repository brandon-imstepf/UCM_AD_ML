import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sympy as sp
from sklearn import metrics
from regression_benchmark_functions import Imstepf_funcs, Nguyen_funcs
from calculate_errors import calculate_mse_nmse
from stat_plots import plot_single_comparison, plot_complexity_vs_error

# Define custom symbolic functions for SymPy parsing
class cube(sp.Function):
    @classmethod
    def eval(cls, x):
        return x**3
    
class square(sp.Function):
    @classmethod
    def eval(cls, x):
        return x**2

# Set global font settings for Matplotlib
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(size=26, family='serif', style='normal')
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})

# Function to create a master folder for storing all graphs
def create_master_folder(base_dir, folder_name="graphs_master"):
    master_folder = os.path.join(base_dir, folder_name)
    os.makedirs(master_folder, exist_ok=True)
    return master_folder

# Function to calculate residuals (absolute difference between true and predicted values)
def calculate_residuals(y_true, y_pred):
    return abs(y_true - y_pred)

# Log residuals to file
def log_residuals(log_folder, file_name, symbolic_eq, residuals):
    log_file_path = os.path.join(log_folder, file_name)
    with open(log_file_path, 'a') as f:
        f.write(f"Equation: {symbolic_eq}\n")
        f.write("Residuals:\n")
        f.write(" ".join(map(str, residuals)) + "\n\n")

# Log differences in residuals
def log_residual_differences(log_folder, in_sample_res, oos_res, symbolic_eq):
    diff_file_path = os.path.join(log_folder, "residual_differences.txt")
    with open(diff_file_path, 'a') as f:
        f.write(f"Equation: {symbolic_eq}\n")
        f.write("Residual Differences:\n")
        f.write(" ".join(map(str, (np.array(in_sample_res) - np.array(oos_res)))) + "\n\n")


def process_all_folders(base_dir):
    """
    Processes all folders in the base directory, computes residuals, errors, and generates graphs 
    comparing true vs. predicted functions from symbolic regression outputs.
    """
    # Create a master folder for storing all generated graphs
    master_folder = create_master_folder(base_dir)
    print("Directories in base_dir:", [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

    print('Processing folders...')
    

    # Iterate through each folder in the base directory
    for noise_folder in os.listdir(base_dir):
        noise_path = os.path.join(base_dir, noise_folder)

        print(f"noise_path = {noise_path}")

        # Skip the master folder or any invalid directories
        if not os.path.isdir(noise_path) or noise_folder == "graphs_master":
            print(f"Skipping invalid or master folder: {noise_folder}")
            continue

        # Process only folders ending with "_Percent_Noise"
        if noise_folder.endswith("_Percent_Noise"):
            print(f"Processing folder: {noise_folder}")
            # Look for a subfolder starting with "pysr_"
            pysr_folder = next((f for f in os.listdir(noise_path) if f.startswith("pysr_")), None)
            if pysr_folder:
                pysr_path = os.path.join(noise_path, pysr_folder)

                # Iterate through each dataset folder in the "pysr_" subfolder
                for dataset_folder in os.listdir(pysr_path):
                    dataset_path = os.path.join(pysr_path, dataset_folder)
                    
                    # Find the main CSV file (excluding "_oos.csv" files)
                    dataset_file = next((f for f in os.listdir(dataset_path) if f.endswith(".csv") and not f.endswith("_oos.csv")), None)
                    if dataset_file:
                        hall_of_fame_path = os.path.join(dataset_path, dataset_file)
                        true_file_path = os.path.join(noise_path, dataset_folder + ".csv")
                        
                        # Check if the corresponding true data CSV file exists
                        if os.path.isfile(true_file_path):
                            # Create a log folder for the dataset
                            dataset_log_folder = os.path.join(master_folder, dataset_folder, "log")
                            os.makedirs(dataset_log_folder, exist_ok=True)
                            
                            # Load the true data (in-sample)
                            true_data = pd.read_csv(true_file_path)
                            x_true = np.sort(true_data.iloc[:, :-1].values, axis=0)
                            y_true = np.sort(true_data.iloc[:, -1].values)

                            # Load out-of-sample data if available
                            oos_file_path = true_file_path.replace(".csv", "_oos.csv")
                            if os.path.isfile(oos_file_path):
                                oos_data = pd.read_csv(oos_file_path)
                                x_oos = np.sort(oos_data.iloc[:, :-1].values, axis=0)
                                y_oos_true = np.sort(oos_data.iloc[:, -1].values)
                            else:
                                oos_data, x_oos, y_oos_true = None, None, None
                            
                            # Load the hall of fame CSV
                            hall_of_fame = pd.read_csv(hall_of_fame_path)
                            complexity = hall_of_fame['Complexity'].values
                            
                            # Initialize lists for error metrics
                            nmse_list, mse_list, oos_nmse_list, oos_mse_list = [], [], [], []
                            
                            # Process each equation in the hall of fame
                            for i, row in hall_of_fame.iterrows():
                                eq_str = row['Equation']
                                try:
                                    # Parse the equation using SymPy
                                    eq = sp.sympify(eq_str, locals={'square': square, 'cube': cube, 'sin': sp.sin, 'cos': sp.cos, 'log': sp.log, 'exp': sp.exp, '^': sp.Pow })
                                    
                                    # Predict values for in-sample data
                                    y_pred = [float(eq.subs(dict(zip([f"x{i}" for i in range(len(x_true[0]))], x))).evalf()) for x in x_true]
                                    in_sample_residuals = calculate_residuals(y_true, y_pred)
                                    log_residuals(dataset_log_folder, "in_sample_residuals.txt", eq_str, in_sample_residuals)
                                    
                                    # Calculate error metrics for in-sample predictions
                                    mse, nmse = calculate_mse_nmse(y_true, y_pred)
                                    nmse_list.append(nmse)
                                    mse_list.append(mse)
                                    
                                    # Process out-of-sample data if available
                                    if oos_data is not None:
                                        y_oos_pred = [float(eq.subs(dict(zip([f"x{i}" for i in range(len(x_oos[0]))], x))).evalf()) for x in x_oos]
                                        oos_residuals = calculate_residuals(y_oos_true, y_oos_pred)
                                        log_residuals(dataset_log_folder, "oos_residuals.txt", eq_str, oos_residuals)
                                        log_residual_differences(dataset_log_folder, in_sample_residuals, oos_residuals, eq_str)
                                        
                                        # Calculate error metrics for out-of-sample predictions
                                        oos_mse, oos_nmse = calculate_mse_nmse(y_oos_true, y_oos_pred)
                                        oos_nmse_list.append(oos_nmse)
                                        oos_mse_list.append(oos_mse)
                                    
                                    # Generate plots for 1D or 2D data
                                    if len(x_true[0]) <= 2:
                                        save_path = os.path.join(master_folder, dataset_folder, f"{dataset_folder}_complexity_{row['Complexity']}.png")
                                        plot_single_comparison(x_true[:, 0], y_true, y_pred, eq_str, row['Complexity'], nmse, mse, save_path)
                                except Exception as e:
                                    print(f"Error processing equation '{eq_str}': {e}")
                                    continue
                            
                            # Generate complexity vs error plots
                            save_path = os.path.join(master_folder, dataset_folder, f"{dataset_folder}_complexity_vs_error.png")
                            plot_complexity_vs_error(complexity, nmse_list, oos_nmse_list, save_path)
                            


# Run the function
base_dir = "C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/Imstepf-RBFs-ns-oos-(3,5,20)"
process_all_folders(base_dir)