import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox

# MATPLOTLIB FONT SETTINGS #
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(size=26, family='serif', style='normal')
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})

def runModel(x, y, func_name, directory, stop_count=5):
    # Change the working directory to the target directory
    os.chdir(directory)

    # Split data into training and testing sets (80/20 split)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Learn equations using PySR
    model = PySRRegressor(
        procs=4,  # Number of processes
        populations=16,  # Number of populations in the evolutionary algorithm
        population_size=250,  # Size of each population
        ncycles_per_iteration=1000,  # Number of cycles per iteration
        niterations=100000,  # Maximum number of iterations
        complexity_of_constants=4,
        parsimony = 0.0001,
        adaptive_parsimony_scaling = 1000,
        #constraints={
        #"/": (-1, 9),
        #"square": 9,
        #"cube": 9,
        #"exp": 9,
        #},
        early_stop_condition=("stop_if(loss,complexity) = loss < 1e-23 && complexity < 15"),  # Early stopping condition
        timeout_in_seconds=60*45,  # Timeout in seconds
        maxsize=30,  # Maximum size of the equations
        maxdepth=5,  # Maximum depth of the equations
        binary_operators=["*", "+", "-", "/","^"],  # Binary operators to be used
        unary_operators=["exp"],  # Unary operators to be used
        #nested_constraints={
        #    "square": {"square": 1, "cube": 0, "exp": 0},
        #    "cube": {"square": 1, "cube": 0, "exp": 0},
        #    "exp": {"square": 1, "cube": 1, "exp": 0},
        #},  # Constraints on nesting unary operators
        #select_k_features=0,  # Number of features to be selected
        progress=False,  # Show progress
        #weight_randomize=2,  # Randomization weight
        #weight_add_node=2,
        weight_optimize=0.001,
        precision=64,  # Precision of the calculations
        warm_start=False,  # Use the previous model as a warm start
        turbo=False,  # Use turbo mode for faster computations
        denoise=False,  # Denoiser (Gaussian white noise kernel)
        model_selection="best",  # Save the best equations
    )

    start_time = datetime.now()
    model.fit(x_train, y_train)

    # Select the best equation based on score and loss
    best_idx = model.equations_.query(f"loss < {2 * model.equations_.loss.min()}").score.idxmax()
    equation = model.equations_.iloc[[best_idx]]

    # Prepare directory for saving results
    equation_dir = os.path.join(directory, f"equation_{best_idx}")
    os.makedirs(equation_dir, exist_ok=True)
    
    current_eq = model.sympy(best_idx)
    y_train_pred = model.predict(x_train, index=best_idx)
    y_test_pred = model.predict(x_test, index=best_idx)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    residuals_train = np.abs(y_train.flatten() - y_train_pred)
    residuals_test = np.abs(y_test.flatten() - y_test_pred)

    # Write equation and timing to log
    with open(os.path.join(equation_dir, "log_file.txt"), "a") as file:
        file.write(f"{func_name}: {current_eq}\n")
        file.write(f"Train MSE: {mse_train}\n")
        file.write(f"Test MSE: {mse_test}\n")

    # Plotting and saving results
    plot_and_save_results(x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, func_name, mse_train, mse_test, equation_dir, residuals_train, residuals_test)

    end_time = datetime.now()
    print(f"Total Duration: {(end_time - start_time).total_seconds()} seconds")

def plot_and_save_results(x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, func_name, mse_train, mse_test, directory, residuals_train, residuals_test):
    """
    Plots and saves the results including residuals on both training and testing data.

    Args:
    - x_train, y_train: Training data and predictions.
    - x_test, y_test: Testing data and predictions.
    - func_name: Name of the function/dataset.
    - mse_train, mse_test: Mean Squared Error for training and testing data.
    - directory: Directory to save the plots.
    - residuals_train, residuals_test: Residuals for training and testing data.
    
    Returns:
    - None
    """

    # Calculate normalized residuals
    residuals_train = np.abs((y_train.reshape(-1) - y_train_pred.reshape(-1)) / y_train.reshape(-1))
    residuals_test = np.abs((y_test.reshape(-1) - y_test_pred.reshape(-1)) / y_test.reshape(-1))

    # Plot residuals for training and testing data
    plt.figure()
    plt.tight_layout()
    plt.scatter(np.abs(y_train), np.abs(residuals_train), alpha=0.75, color='b', label="Train Residuals")
    plt.scatter(np.abs(y_test), np.abs(residuals_test), alpha=0.75, color='r', label="Test Residuals")
    plt.xlabel("Normalized Abs True Values")
    plt.ylabel("Normalized Abs Residuals")
    plt.title(f"Residuals for {func_name} (Norm. Abs. Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(directory, f"{func_name}_residuals_abs.png"))
    plt.close()

def process_all_datasets(base_directory):
    """
    Processes all datasets in the specified base directory.

    Args:
    base_directory: The directory containing the datasets.
    """
    csv_files = [f for f in os.listdir(base_directory) if f.endswith('.csv')]

    for file in csv_files:
        df = pd.read_csv(os.path.join(base_directory, file))
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)
        func_name = file.split('.')[0]

        # Create a sub-directory for each dataset to save its results
        dataset_dir = os.path.join(base_directory, func_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Run the model on the dataset
        runModel(x, y, func_name, dataset_dir)


base_directory = "C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/Sept_20_Cumulative"
process_all_datasets(base_directory)
