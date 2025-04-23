import os
import pandas as pd
import sympy as sp
from regression_benchmark_functions import *  # Import benchmark functions
import inspect
from dataset_plots import *

def extract_hall_of_fame_equations(csv_path):
    """Extracts equations from the hall of fame CSV file."""
    df = pd.read_csv(csv_path)
    return df[['Complexity', 'Loss', 'Equation']]

def match_dataset_to_function(dataset_name):
    """Returns the correct function for a given dataset name."""
    function_map = {
        'Nguyen_F1': Nguyen_F1, 'Nguyen_F2': Nguyen_F2, 'Nguyen_F3': Nguyen_F3, 'Nguyen_F4': Nguyen_F4,
        'Nguyen_F5': Nguyen_F5, 'Nguyen_F6': Nguyen_F6, 'Nguyen_F7': Nguyen_F7, 'Nguyen_F8': Nguyen_F8,
        'Nguyen_F9': Nguyen_F9, 'Nguyen_F10': Nguyen_F10,
        'Imstepf_F1': Imstepf_F1, 'Imstepf_F2': Imstepf_F2, 'Imstepf_F3': Imstepf_F3, 'Imstepf_F4': Imstepf_F4
    }
    return function_map.get(dataset_name, None)

def compute_accuracy(predicted_expr, correct_expr):
    """Computes accuracy as a percentage based on term-wise correctness."""
    predicted_terms = set(sp.expand(predicted_expr).as_ordered_terms())
    correct_terms = set(sp.expand(correct_expr).as_ordered_terms())
    
    matched_terms = len(predicted_terms.intersection(correct_terms))
    total_terms = len(correct_terms)
    
    return (matched_terms / total_terms) * 100 if total_terms > 0 else 0

def process_directory(base_dir,troubleshoot_boolean):
    """Processes the given base directory and extracts results."""
    results = []
    noise_levels = [d for d in os.listdir(base_dir) if d.endswith('_Percent_Noise')]

    if troubleshoot_boolean:
        print(f"Noise Levels: {noise_levels}")
    
    for noise_level in noise_levels:
        noise_path = os.path.join(base_dir, noise_level)

        if troubleshoot_boolean:
            print(f"Noise Path: {noise_path}")

        pysr_folder = next((f for f in os.listdir(noise_path) if f.startswith('pysr_')), None)
        if not pysr_folder:
            print("Did you remember to create the pysr_ folder?\n")
            continue
        
        pysr_path = os.path.join(noise_path, pysr_folder)

        if troubleshoot_boolean:
            print(f"PySR Path: {pysr_path}")

        for dataset_folder in os.listdir(pysr_path):
            dataset_path = os.path.join(pysr_path, dataset_folder)

            if troubleshoot_boolean:
                print(f"Dataset Path: {dataset_path}")

            csv_name = next((f for f in os.listdir(dataset_path) if f.endswith('.csv')), None)

            if troubleshoot_boolean:
                print(f"CSV Name: {csv_name}")

            if not csv_name:
                continue

            hof_csv_path = os.path.join(dataset_path, csv_name)

            if troubleshoot_boolean:
                print(f"Hall of Fame CSV Path: {hof_csv_path}")

            base_name = "_".join(dataset_folder.split("_")[:2])

            if troubleshoot_boolean:
                print(f"Base Name: {base_name}")

            correct_function = match_dataset_to_function(base_name)
            if not correct_function:
                continue
            
            # Convert correct function to SymPy expression
            x_symbols = sp.symbols(f'x0:{correct_function.__code__.co_argcount}')

            # Map NumPy functions to SymPy functions
            np_to_sympy = {'np.sin': 'sp.sin', 'np.cos': 'sp.cos', 'np.exp': 'sp.exp', 'np.log': 'sp.log'}

            # Get function source code
            func_source = inspect.getsource(correct_function)

            # Replace NumPy functions with SymPy functions in the source code
            for np_func, sp_func in np_to_sympy.items():
                func_source = func_source.replace(np_func, sp_func)

            # Execute modified function definition
            exec(func_source, globals(), locals())

            # Get the new function without NumPy
            correct_function_sympy = locals()[correct_function.__name__]

            # Evaluate with SymPy symbols
            correct_expr = correct_function_sympy(*x_symbols)

            # Ensure the result is a valid SymPy expression
            correct_expr = sp.sympify(correct_expr)

            print(f"Converted SymPy expression: {correct_expr}")

            
            hof_equations = extract_hall_of_fame_equations(hof_csv_path)
            
            for _, row in hof_equations.iterrows():
                try:
                    predicted_expr = sp.sympify(row['Equation'], locals={f'x{i}': x_symbols[i] for i in range(len(x_symbols))})
                    accuracy = compute_accuracy(predicted_expr, correct_expr)
                    
                    results.append({
                        'Dataset': dataset_folder,
                        'Noise Level': noise_level,
                        'Matched Equation': row['Equation'],
                        'Complexity': row['Complexity'],
                        'Loss': row['Loss'],
                        'Accuracy (%)': accuracy
                    })
                    
                    if accuracy == 100:
                        break  # Stop if perfect match found
                except Exception:
                    continue  # Skip invalid equations
    
    return pd.DataFrame(results)

runProcess = True

# Example Usage:
base_directory = r"C:\Users\brand\Documents\Python Scripts\pysr_code\datasets\Imstepf-RBFs-n-ns-(1,3,40)"
os.chdir(base_directory)
csv_name = 'equation_matching_results.csv'

if runProcess:
    results_df = process_directory(base_directory,troubleshoot_boolean=False)
    results_df.to_csv(csv_name, index=False)
    print(results_df)

results_df = pd.read_csv(csv_name)

# Group datasets by function (F1, F2, etc.)
grouped_datasets = {}
for _, row in results_df.iterrows():
    function_name = row['Dataset'].split('_')[0] + '_' + row['Dataset'].split('_')[1]
    if function_name not in grouped_datasets:
        grouped_datasets[function_name] = []
    grouped_datasets[function_name].append(row)

# Convert grouped datasets to DataFrames
grouped_dataframes = {k: pd.DataFrame(v) for k, v in grouped_datasets.items()}

# Display a table representing the grouped_dataframes for troubleshooting purposes
for key, df in grouped_dataframes.items():
    print(f"Function: {key}")
    print(df)

# Plot the accuracy vs. noise level for each function
for key, df in grouped_dataframes.items():
    plot_accuracy_vs_noise3(df)


#plot_line_accuracy(results_df)
#plot_accuracy_vs_noise_2(results_df)
#plot_accuracy_vs_noise3(results_df)
#average_accuracy_vs_dataset(results_df['Function'])