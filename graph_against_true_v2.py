import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
To rewrite:
    Use Pandas to open csv's ..
    use SymPy to read the functions inside the csv's ..
    use RegEx
        ^hall_of_fame_.*\.csv 
        matches
        hall_of_fame_2024-10-07_124744.825.csv (and everything else that starts with hall_of_fame and ends with .csv)

    
"""

# Function to clean the symbolic equations
def clean_equation(equation):
    """
    Cleans the equation by replacing incorrect operators and handling potential issues.
    """
    equation = equation.replace("^", "**")  # Replace bitwise XOR with exponentiation
    return equation

# Function to safely evaluate the equations
def evaluate_function(equation, inputs):
    """
    Safely evaluate the symbolic equation with multiple inputs (x0, x1, x2, ..., x5).
    inputs: A list of arrays, each representing one input variable.
    """
    equation = clean_equation(equation)
    
    try:
        for i in range(len(inputs)):
            equation = equation.replace(f"x{i}", f"inputs[{i}]")
        
        # Evaluate the equation safely using eval
        return eval(equation)
    except ZeroDivisionError:
        print(f"Warning: Division by zero in function {equation}")
        return np.full_like(inputs[0], np.nan)
    except Exception as e:
        print(f"Error evaluating function {equation}: {e}")
        return np.full_like(inputs[0], np.nan)

# Function to calculate errors (MSE, NMSE, Absolute Error)
def calculate_errors(pred_values, true_values):
    """
    Calculate MSE, NMSE, and absolute error between predicted and true values.
    """
    mse = np.mean((pred_values - true_values) ** 2)
    nmse = mse / np.mean(true_values ** 2)
    absolute_error = np.abs(pred_values - true_values)
    return mse, nmse, absolute_error

# Function to find the 'hall of fame' CSV files and match them to the true value files
def get_csv_files(root_dir):
    """
    Recursively finds all 'hall_of_fame' CSV files within the subdirectories of root_dir and matches them with the true value files in the root directory.
    
    Returns:
    - A dictionary where keys are folder names and values are tuples with hall_of_fame CSV path and corresponding true value CSV path.
    """
    folder_csv_map = {}
    
    for folder_name, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith("hall_of_fame") and file.endswith(".csv"):
                # Folder name without root_dir prefix
                folder_relative = os.path.relpath(folder_name, root_dir)
                # Corresponding true value CSV in the root directory
                base_folder_name = folder_relative.split(os.sep)[-1]
                true_value_csv = os.path.join(root_dir, f"{base_folder_name}.csv")
                
                if os.path.exists(true_value_csv):
                    hall_of_fame_csv = os.path.join(folder_name, file)
                    folder_csv_map[base_folder_name] = (hall_of_fame_csv, true_value_csv)
    
    return folder_csv_map

# Function to process hall of fame CSVs and plot/save residual results
def process_hall_of_fame_csv(hall_of_fame_csv, true_value_csv, output_dir):
    """
    Process the hall of fame CSV to calculate residuals and generate plots.
    Each function's residuals are plotted separately against the true values and saved to the output directory.
    """
    # Load hall of fame and true value data
    hall_of_fame_data = pd.read_csv(hall_of_fame_csv)
    true_value_data = pd.read_csv(true_value_csv)
    
    # Extract input columns and true values
    x_values = [true_value_data.iloc[:, i].values for i in range(true_value_data.shape[1] - 1)]  # Inputs (x0, x1, ..., x5)
    true_values = true_value_data.iloc[:, -1].values  # True solution (last column)

    errors_summary = []
    
    # Create output directory for plots if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each equation in the hall of fame CSV
    for idx, row in hall_of_fame_data.iterrows():
        complexity = row['Complexity']
        loss = row['Loss']
        equation = row['Equation']
        
        # Evaluate the symbolic equation over the inputs
        pred_values = evaluate_function(equation, x_values)
        
        # Calculate errors
        mse, nmse, abs_error = calculate_errors(pred_values, true_values)
        residuals = pred_values - true_values  # Calculate residuals

        # Store results for output CSV
        errors_summary.append({
            'Complexity': complexity,
            'Loss': loss,
            'MSE': mse,
            'NMSE': nmse,
            'Absolute Error': np.mean(abs_error)
        })
        
        # Plot residuals vs. domain (x0)
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values[0], residuals, label=f'Residuals (Complexity {complexity})')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.legend()
        
        # Add error metrics and complexity to the plot
        plt.title(f"Residuals for Equation {idx} (Complexity {complexity})")
        plt.xlabel("Domain (x0)")
        plt.ylabel("Residuals (Predicted - True)")
        plt.text(0.95, 0.95, f"Complexity: {complexity}\nMSE: {mse:.4f}\nNMSE: {nmse:.4f}\nAbs Error: {np.mean(abs_error):.4f}",
                 verticalalignment='top', horizontalalignment='right',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7))
        
        # Save the plot in the output directory
        plot_filename = os.path.join(output_dir, f'Residuals_Equation_{idx}_Complexity_{complexity}.png')
        plt.savefig(plot_filename)
        plt.close()

    # Convert the errors summary into a DataFrame for saving
    errors_df = pd.DataFrame(errors_summary)
    
    # Save the output CSV (errors summary)
    output_csv_path = os.path.join(output_dir, 'error_summary.csv')
    errors_df.to_csv(output_csv_path, index=False)
    
    return errors_df

# change root dir here
root_dir = "C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/RBFs-n-oos-(-1,1,20)"
root_output_dir = os.path.join(root_dir, 'hall_of_fame_results')

# Get the matched hall_of_fame and true_value CSV files
folder_csv_map = get_csv_files(root_dir)

# Process each folder's hall_of_fame CSV and generate results
for folder, (hall_of_fame_csv, true_value_csv) in folder_csv_map.items():
    folder_output_dir = os.path.join(root_output_dir, folder)
    process_hall_of_fame_csv(hall_of_fame_csv, true_value_csv, folder_output_dir)
