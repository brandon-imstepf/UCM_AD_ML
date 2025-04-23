# Script to call functions for use on data created by pysr_train.pysr_train
import pandas as pd
import os
import matplotlib as plt
from stat_plots import *
from sklearn.metrics import r2_score

# MATPLOTLIB FONT SETTINGS #
from matplotlib.font_manager import FontProperties
font_prop = FontProperties(size=26, family='serif', style='normal')
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({
    'font.size': 14,
    'lines.color': 'orange',
    'scatter.edgecolors': '#1F77B4'
})
# ------------------------ #

# Establish Directories #
base_directory = 'C:/Users/brand/Desktop/Raj-Sindi/training_data/sim_csv_v7_w/1-9 frac07/brand_1-9_flux_and_w1'
info_directory = base_directory + "/Info/"
os.chdir(base_directory)

# Find "best" function from best.txt"
with open(info_directory + "best.txt", "r") as file:
    first_line = file.readline()
# Extract number
start_text = "Selected Best Function: "
best_idx = int( first_line[len(start_text):].split(",")[0].strip() )

# Read .csv's #
model_equations = pd.read_csv(info_directory + 'model_equations.csv')
test_datalog_primary = pd.read_csv(info_directory + 'test_datalog_primary.csv')
train_datalog_primary = pd.read_csv(info_directory + 'train_datalog_primary.csv')
validation_datalog_primary = pd.read_csv(info_directory + 'validation_datalog_primary.csv')
test_datalog_secondary = pd.read_csv(info_directory + 'test_datalog_secondary.csv')
train_datalog_secondary = pd.read_csv(info_directory + 'train_datalog_secondary.csv')
validation_datalog_secondary = pd.read_csv(info_directory + 'validation_datalog_secondary.csv')


visualization_dir = os.path.join(info_directory, "Visualizations")
os.makedirs(visualization_dir, exist_ok=True)

# Plot complexity vs diversity #
plot_complexity_vs_diversity(model_equations['equation'], model_equations['complexity'],best=best_idx)
# Training Graphs #


# Calculate diversity for each equation
diversity = []
for eq in model_equations['equation']:
    # Parse the equation into a SymPy object
    sympy_eq = sp.sympify(eq)
    # Find unique variables used in the equation
    unique_vars = sympy_eq.free_symbols
    # Count the number of unique variables
    diversity.append(len(unique_vars))





'''
# Compare Residuals#
residual_list = []
residual_list.append(validation_datalog_primary['Model_13_Residuals'])
residual_list.append(validation_datalog_primary['Model_17_Residuals'])

models_of_interest = ["Model_13","Model_17"]
filtered_data = validation_datalog_secondary[validation_datalog_secondary['Model'].isin(models_of_interest)]
nmse_values = filtered_data.set_index('Model')['NMSE_Validation'].to_dict()


'''
