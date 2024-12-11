import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from regression_benchmark_functions import Imstepf_funcs

current_func = 1

# Load CSV file
basedir = 'C:/Users/brand/Documents/Python Scripts/pysr_code/datasets/Imstepf_RBFs-oos-(0,1,20)/Sept_30/Imstepf_F1_LHS/'
csv_file = basedir + 'hall_of_fame_2024-09-30_140122.490.csv'
df = pd.read_csv(csv_file)

# Define a function to evaluate each equation as a lambda function
def equation_to_lambda(eq_str):
    # Replace exp with np.exp for compatibility with numpy
    eq_str = eq_str.replace("exp", "np.exp")
    for i in range(5):
        eq_str = eq_str.replace(f"x{i}", f"x[{i}]")
    return eval(f"lambda x: {eq_str}")

# Create a domain to evaluate the functions
domain = np.linspace(0, 1, 20)
X = np.array(np.meshgrid(domain, domain, domain, domain, domain)).T.reshape(-1, 5)

# Prepare data for scatter matrix
scatter_data = pd.DataFrame()

for idx, row in df.iterrows():
    equation = equation_to_lambda(row['Equation'])
    benchmark_function = Imstepf_funcs[current_func]  # Select appropriate benchmark function
    
    # Evaluate both functions on the domain
    y_symbolic = np.array([equation(x) for x in X])
    y_benchmark = np.array([benchmark_function(*x) for x in X])

    # Add both functions' evaluations to the dataframe for scatter matrix
    scatter_data[f'Symbolic Eq {idx} (C={row["Complexity"]})'] = y_symbolic
    scatter_data[f'Benchmark {current_func}'] = y_benchmark

# Plot scatter matrix with diagonal KDE for both symbolic and benchmark functions
plt.figure(figsize=(10, 8))
scatter_matrix(scatter_data, alpha=0.2, figsize=(10, 10), diagonal='kde', color=['blue', 'gold'])
plt.suptitle('Symbolic vs Benchmark Functions Comparison')
plt.show()
