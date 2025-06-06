import os
import pandas as pd
import sympy as sp
from regression_benchmark_functions import *  # Import benchmark functions
import inspect
from dataset_plots import *
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

#Change matplotlib settings to enlarge text and use Helvetica font
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 14})

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
            #print("Did you remember to create the pysr_ folder?\n")
            # Create the pysr_ folder if it doesn't exist, using date and time
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            pysr_folder = f"pysr_{timestamp}"
            os.makedirs(os.path.join(noise_path, pysr_folder), exist_ok=True)
            # Then, put all "Nguyen_" or "Imstepf_" folders in the pysr_ folder, but not .csv files
            for folder in os.listdir(noise_path):
                if folder.startswith('Nguyen_') or folder.startswith('Imstepf_'):
                    folder_path = os.path.join(noise_path, folder)
                    if os.path.isdir(folder_path):
                        new_folder_path = os.path.join(noise_path, pysr_folder, folder)
                        os.rename(folder_path, new_folder_path)
                        print(f"Moved folder: {folder_path} to {new_folder_path}")

            print(f"Created new folder: {pysr_folder}")


        pysr_folder = next((f for f in os.listdir(noise_path) if f.startswith('pysr_')), None)
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
            np_to_sympy = {'np.sin': 'sp.sin', 'np.cos': 'sp.cos', 'np.exp': 'sp.exp', 'np.log': 'sp.log', 'np.sqrt': 'sp.sqrt'}

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

# Set the base directory to the location of your datasets
base_directory = r"C:\Users\brand\Documents\Python Scripts\pysr_code\datasets\Nguyen-RBFs-n-ns-(1,3,40)\LEARNED"
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
#for key, df in grouped_dataframes.items():
    #plot_accuracy_vs_noise3(df)


#plot_line_accuracy(results_df)
#plot_accuracy_vs_noise_2(results_df)
#plot_accuracy_vs_noise3(results_df)
#average_accuracy_vs_dataset(results_df['Function'])

def extract_function_id(dataset_name):
    """Extracts F1, F2, etc. from dataset like 'Nguyen_F1_LHS'"""
    for part in dataset_name.split("_"):
        if part.startswith("F") and part[1:].isdigit():
            return part
    return dataset_name  # fallback

def extract_noise_label(noise_str):
    """Extracts numeric part and formats as 'N%'"""
    return f"{noise_str.split('_')[0]}%"

def sort_noise_key(noise_str):
    """Sorts based on numeric part of noise level string"""
    return int(noise_str.split('_')[0])

def plot_summary_accuracy_LHS(results_df):
    # Filter for LHS sampling
    lhs_df = results_df[results_df["Sampling Method"] == "LHS"].copy()

    # Add simplified function and noise labels
    lhs_df["Function"] = lhs_df["Dataset"].apply(extract_function_id)
    lhs_df["Noise Label"] = lhs_df["Noise Level"].apply(extract_noise_label)

    # Compute best accuracy for each function at each noise level
    grouped = lhs_df.groupby(["Function", "Noise Label"])["Accuracy (%)"].max().reset_index()

    # Sort functions numerically (F1, F2, ..., F10)
    grouped["Function_Num"] = grouped["Function"].str.extract(r'F(\d+)').astype(int)
    grouped["Noise Sort"] = grouped["Noise Label"].str.rstrip('%').astype(int)
    grouped = grouped.sort_values(["Function_Num", "Noise Sort"])

    # Set up plot
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    functions = sorted(grouped["Function"].unique(), key=lambda x: int(x[1:]))
    for func in functions:
        func_data = grouped[grouped["Function"] == func]
        plt.plot(
            func_data["Noise Label"],
            func_data["Accuracy (%)"],
            label=func,
            linewidth=2.5
        )

    plt.title("Accuracy vs Noise Level")
    plt.xlabel("Noise Level")
    plt.ylabel("Best Accuracy (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Legend below the plot, sorted correctly
    '''
    plt.legend(
        title="Function",
        loc='upper right',
        bbox_to_anchor=(0.5, -0.15),
        #ncol=6,
        frameon=True
    )
    '''
    plt.legend()
    plt.show()


# Extract sampling method from the dataset name
def get_sampling_method(name):
    parts = name.split('_')
    return parts[-1]  # Last part is 'LHS', 'Uniform', or 'RandSample'

results_df['Sampling Method'] = results_df['Dataset'].apply(get_sampling_method)

# Find the best (max) accuracy per dataset
max_accuracy_per_dataset = results_df.groupby('Dataset')['Accuracy (%)'].max().reset_index()

# Add sampling method info back in
max_accuracy_per_dataset['Sampling Method'] = max_accuracy_per_dataset['Dataset'].apply(get_sampling_method)

# Now group by sampling method and compute average of those best scores
sampling_method_scores = max_accuracy_per_dataset.groupby('Sampling Method')['Accuracy (%)'].mean().sort_values(ascending=False)

print("Average of Best Accuracies per Sampling Method:")
print(sampling_method_scores)

plt.figure(figsize=(8, 6))
sampling_method_scores.plot(kind='bar', color='royalblue')
plt.ylabel('Average of Best Accuracy (%)')
plt.title('Average of Best Accuracy per Sampling Method')
plt.xticks(rotation=0)
#plt.grid(axis='y')
plt.tight_layout()
plt.show()


import re

# Step 1: Filter only LHS
lhs_df = results_df[results_df['Sampling Method'] == 'LHS'].copy()

# Step 2: Extract function name (e.g., 'Nguyen_F2' from 'Nguyen_F2_LHS')
lhs_df['Function'] = lhs_df['Dataset'].apply(lambda x: '_'.join(x.split('_')[:-1]))

# Step 3: Group and get max accuracy per function
max_accuracy = lhs_df.groupby('Function')['Accuracy (%)'].max().reset_index()

# Step 4: Extract function number for proper sorting
def extract_function_number(name):
    match = re.search(r'F(\d+)', name)
    return int(match.group(1)) if match else float('inf')

max_accuracy['FunctionNumber'] = max_accuracy['Function'].apply(extract_function_number)
max_accuracy_sorted = max_accuracy.sort_values('FunctionNumber')

# Step 5: Plot
plt.figure(figsize=(10, 6))
plt.bar(max_accuracy_sorted['Function'], max_accuracy_sorted['Accuracy (%)'], color='forestgreen')
plt.ylabel('Max Accuracy (%)')
plt.title('Best Accuracy per Function using LHS Sampling')
plt.xticks(rotation=45, ha='right')
#plt.grid(axis='y')
plt.tight_layout()
plt.show()

plot_summary_accuracy_LHS(results_df)