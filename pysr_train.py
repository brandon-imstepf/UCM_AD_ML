import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.colors as mcolors

#troubleshooting#
import tkinter as tk
from tkinter import messagebox
#################

# MATPLOTLIB FONT SETTINGS #
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(size=26, family='serif', style='normal')
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})
#######################


############################
# to graph all spat equations, set true #
run_all_equations = False
############################

def runModel(x, y, func_name, directory, noise_level=None, stop_count=5, run_all_equations=False):
    # Change the working directory to the target directory
    os.chdir(directory)

    # Split data into training and testing sets (80/20 split)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


    # Learn equations using PySR
    model = PySRRegressor(
        procs=4,  # Number of processes
        populations=8,  # Number of populations in the evolutionary algorithm
        population_size=50,  # Size of each population
        ncycles_per_iteration=500,  # Number of cycles per iteration
        niterations=100000,  # Maximum number of iterations
        early_stop_condition=("stop_if(loss,complexity) = loss < 1e-6 && complexity < 15"),  # Early stopping condition
        timeout_in_seconds=60*10,  # Timeout in seconds
        maxsize=30,  # Maximum size of the equations
        maxdepth=5,  # Maximum depth of the equations
        binary_operators=["*", "+", "-", "/"],  # Binary operators to be used
        unary_operators=["square", "cube", "exp", "sin", "cos", "log"],  # Unary operators to be used
        nested_constraints={
            "square": {"square": 1, "cube": 0, "exp": 0},
            "cube": {"square": 1, "cube": 0, "exp": 0},
            "exp": {"square": 1, "cube": 1, "exp": 0},
        },  # Constraints on nesting unary operators
        select_k_features=7,  # Number of features to be selected
        progress=True,  # Show progress
        weight_randomize=0.1,  # Randomization weight
        precision=64,  # Precision of the calculations
        warm_start=True,  # Use the previous model as a warm start
        turbo=True,  # Use turbo mode for faster computations
        denoise=True, #denoiser (gaussian white noise kernel)
        model_selection="best",  # Save the best equations
    )

    start_time = datetime.now()
    model.fit(x_train, y_train)
    
    if run_all_equations:
        equations = model.equations_
    else:
        best_idx = model.equations_.query(f"loss < {2 * model.equations_.loss.min()}").score.idxmax()
        equations = model.equations_.iloc[[best_idx]]

    for idx, equation in equations.iterrows():
        # Prepare directory for saving results for this equation
        equation_dir = os.path.join(directory, f"equation_{idx}") if run_all_equations else directory
        if not os.path.exists(equation_dir):
            os.makedirs(equation_dir)
        
        current_eq = model.sympy(idx)
        y_train_pred = model.predict(x_train, index=idx)
        y_test_pred = model.predict(x_test, index=idx)
        
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        residuals_train = np.abs(y_train.flatten() - y_train_pred)
        residuals_test = np.abs(y_test.flatten() - y_test_pred)

        # Write equation, noise level, and timing to log
        with open(os.path.join(equation_dir, "log_file.txt"), "a") as file:
            file.write(f"{func_name}: {current_eq}\n")
            file.write(f"Train MSE: {mse_train}\n")
            file.write(f"Test MSE: {mse_test}\n")
            if noise_level is not None:
                file.write(f"Noise Level: {noise_level}\n")
            #file.write(f"Train Residuals: {residuals_train}\n")
            #file.write(f"Test Residuals: {residuals_test}\n")

        # Plotting and saving results
        plot_and_save_results(x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, func_name, mse_train, mse_test, noise_level, equation_dir, residuals_train, residuals_test)

    end_time = datetime.now()
    print(f"Total Duration: {(end_time - start_time).total_seconds()} seconds")

def plot_and_save_results(x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, func_name, mse_train, mse_test, noise_level, directory, residuals_train, residuals_test):
    """
    Plots and saves the results including residuals on both training and testing data.

    Args:
    - x_train, y_train: Training data and predictions.
    - x_test, y_test: Testing data and predictions.
    - func_name: Name of the function/dataset.
    - mse_train, mse_test: Mean Squared Error for training and testing data.
    - noise_level: Optional noise level of the data.
    - directory: Directory to save the plots.
    - residuals_train, residuals_test: Residuals for training and testing data.
    
    Returns:
    - None
    """
    num_dims = x_train.shape[1]
    noise_info = f"Noise Level: {noise_level}" if noise_level is not None else ""

    if num_dims == 1:
        plt.tight_layout()
        plt.scatter(x_train, y_train, alpha=0.75, label="Training Data", color='blue')
        plt.scatter(x_train, y_train_pred, alpha=0.75, label="Predicted Training Data", color='cyan')
        plt.scatter(x_test, y_test, alpha=0.75, label="Testing Data", color='red')
        plt.scatter(x_test, y_test_pred, alpha=0.75, label="Predicted Testing Data", color='gold')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title(f"Function {func_name}, \n{noise_info}")
        # Get the current axis
        ax = plt.gca()

        # Add MSE information at the bottom
        plt.text(0.5, -0.15, f"Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=10)

        # Adjust the bottom margin to make room for the text
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(directory, f"{func_name}.png"))
        plt.close()

    elif num_dims == 2:
        fig = plt.figure()
        plt.tight_layout()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], y_train, alpha=0.2, label="Training Data", color='blue')
        ax.scatter(x_train[:, 0], x_train[:, 1], y_train_pred, alpha=0.2, label="Predicted Training Data", color='slateblue')
        ax.scatter(x_test[:, 0], x_test[:, 1], y_test, alpha=0.2, label="Testing Data", color='red')
        ax.scatter(x_test[:, 0], x_test[:, 1], y_test_pred, alpha=0.2, label="Predicted Testing Data", color='gold')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        ax.legend()
        ax.set_title(f"Function {func_name}, Train MSE: {mse_train}, Test MSE: {mse_test}\n{noise_info}")
        plt.savefig(os.path.join(directory, f"{func_name}.png"))
        plt.close()

    # Ensure residuals are calculated correctly
    #residuals_train = np.abs(y_train.reshape(-1) - y_train_pred.reshape(-1))
    #residuals_test = np.abs(y_test.reshape(-1) - y_test_pred.reshape(-1))

    residuals_train = np.abs( (y_train.reshape(-1) - y_train_pred.reshape(-1))/y_train.reshape(-1))
    residuals_test = np.abs( (y_test.reshape(-1) - y_test_pred.reshape(-1))/y_test.reshape(-1))

    """"
    # Plot residuals for training and testing data
    plt.figure()
    plt.tight_layout()
    plt.scatter(np.log10(y_train), np.log10(residuals_train), alpha=0.75, color='b', label="Train Residuals")
    plt.scatter(np.log10(y_test), np.log10(residuals_test), alpha=0.75, color='r', label="Test Residuals")
    plt.xlabel("Log True Values")
    plt.ylabel("Log Residuals")
    plt.title(f"Residuals for {func_name} (Log-Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(directory, f"{func_name}_residuals_loglog.png"))
    plt.close()
    """
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


def read_dataset_info(info_file_path):
    """
    Reads noise level information from a dataset_info.txt file.
    
    Args:
    info_file_path: Path to the dataset_info.txt file.
    
    Returns:
    noise_data: Dictionary mapping function names to their noise levels.
    """
    noise_data = {}
    with open(info_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'noise_std =' in line:
                parts = line.strip().split(',')
                func_name = parts[0].strip()
                noise_std = float(parts[-1].split('=')[-1].strip())
                noise_data[func_name] = noise_std
    return noise_data


def process_csv_files(csv_directory):
    """
    Processes all CSV files in a directory, runs the model, and logs the results.
    
    Args:
    csv_directory: Directory containing CSV files.
    """
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    dataset_info_path = os.path.join(csv_directory, "dataset_info.txt")
    noise_data = read_dataset_info(dataset_info_path) if os.path.exists(dataset_info_path) else {}

    for file in csv_files:
        df = pd.read_csv(os.path.join(csv_directory, file))
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)
        func_name = file.split('.')[0]
        noise_level = noise_data.get(func_name, None)
        
        # Perform 80/20 train/test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        os.makedirs(csv_directory + file)
        
        # Run the model on the training data
        runModel(x_train, y_train, func_name, directory, noise_level=noise_level)


# Create a folder named with the current date and time
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
basedir = 'C:/Users/brand/Documents/Python Scripts/pysr_code'
csv_directory = os.path.join(basedir + '/datasets/2-2-2rand,std tp/individual_trimmed')  # Folder with CSV files
directory = os.path.join(basedir, current_time)
os.makedirs(directory)

# Process CSV files and run the models
process_csv_files(csv_directory)

""""
# a message when it's all done
root = tk.Tk()
root.withdraw()  # Hide the main window
messagebox.showinfo("ALL DONE", "COME BACK TO THE COMPUTER!!!!")
root.bell()
root.destroy()
"""