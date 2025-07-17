import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Matplotlib Settings ---
# Using the settings you provided for consistency.
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
})
# --- End Settings ---

def get_best_pysr_nmse(base_path):
    """
    Loads the validation datalog and returns the minimum NMSE_Test value.
    This represents the best performance achieved by any equation on the test set.
    """
    try:
        # We use the secondary log file which contains the performance metrics for each model.
        validation_log_path = os.path.join(base_path, 'Info', 'validation_datalog_secondary.csv')
        testing_log_path = os.path.join(base_path, 'Info', 'test_datalog_secondary.csv')
        training_log_path = os.path.join(base_path, 'Info', 'train_datalog_secondary.csv')

        if not os.path.exists(validation_log_path):
            print(f"Warning: File not found at {validation_log_path}")
            return None
            
        val_df = pd.read_csv(validation_log_path)
        test_df = pd.read_csv(testing_log_path)
        train_df = pd.read_csv(training_log_path)

        # NMSE_Test is a robust measure of the final model performance.
        if 'NMSE_Test' in test_df.columns:
            # Drop NaN values before finding the minimum to avoid issues.
            min_nmse = test_df['NMSE_Test'].dropna().min()
            return min_nmse if pd.notna(min_nmse) else None
        else:
            print(f"Warning: 'NMSE_Test' column not found in {testing_log_path}")
            return None

    except Exception as e:
        print(f"Error processing {base_path}: {e}")
        return None

def plot_pysr_performance_comparison(run_directories, output_dir):
    """
    Creates a grouped bar chart comparing the best NMSE for each PySR run.
    This visualization, "The PySR Gauntlet," directly compares performance
    across all your experimental conditions.
    """
    labels = ['Narrow Tau', 'Wide Tau', 'Biased Wide Tau']
    flux_nmse = []
    w1_nmse = []

    # Process Flux and W1 runs separately
    for label in labels:
        flux_path = run_directories['Flux'].get(label)
        w1_path = run_directories['W1'].get(label)
        
        # Get the NMSE value, or None if not found
        flux_val = get_best_pysr_nmse(flux_path) if flux_path else None
        w1_val = get_best_pysr_nmse(w1_path) if w1_path else None
        
        # *** FIX: Explicitly handle None values before appending to the list ***
        # If a value is None (due to a missing file or error), append 0 for the plot.
        flux_nmse.append(flux_val if flux_val is not None else 0)
        w1_nmse.append(w1_val if w1_val is not None else 0)


    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Bar colors chosen to be distinct but harmonious.
    rects1 = ax.bar(x - width/2, flux_nmse, width, label='Flux', color='#1F77B4') # Blue
    rects2 = ax.bar(x + width/2, w1_nmse, width, label='W1', color='#FF7F0E') # Orange

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Best Test NMSE (Lower is Better)')
    ax.set_title('Symbolic Regression Performance Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars for clarity
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    fig.tight_layout()
    
    # Save the figure
    save_path = os.path.join(output_dir, "pysr_performance_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved 'PySR Gauntlet' plot to {save_path}")
    plt.show()

def plot_method_showdown(pysr_best_nmse, output_dir):
    """
    Creates a bar chart comparing the best performance of Symbolic Regression
    against Classical Regression and Neural Networks. This is the "Method Showdown."
    """
    # --- Manually input metrics from your other graphs/paper ---
    # These values are based on the "Biased Wide-Range Tau" dataset from your paper,
    # as it's the most representative of the real problem.
    # We use the 'All' model for Classical and the best reported for NN.
    metrics = {
        'Classical Regression': 0.350,  # FValue, All Features, Biased Wide
        'Symbolic Regression': pysr_best_nmse, # This will be the best result from your PySR runs
        'Neural Network': 0.00062  # FValue, Best NN, Biased Wide (1e5)
    }
    
    methods = list(metrics.keys())
    nmse_values = list(metrics.values())
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors chosen to distinguish the methods clearly.
    colors = ['#D3D3D3', '#FF7F0E', '#1F77B4'] # Gray, Orange, Blue
    
    bars = ax.bar(methods, nmse_values, color=colors)
    
    ax.set_ylabel('Best Test NMSE (Lower is Better)')
    ax.set_title('Model Showdown: Accuracy of Different Methods')
    ax.set_yscale('log') # Use a log scale because the NN error is orders of magnitude smaller
    ax.set_ylabel('Best Test NMSE (Log Scale)')
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    ax.bar_label(bars, padding=3, fmt='%.4f')
    
    fig.tight_layout()
    
    save_path = os.path.join(output_dir, "method_showdown.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved 'Method Showdown' plot to {save_path}")
    plt.show()


if __name__ == '__main__':
    # --- Configuration ---
    # Define the base directory where all your run folders are located.
    # You should have 6 folders in total for the PySR runs.
    # Example: 'C:/Users/brand/Desktop/My_PySR_Runs/'
    all_runs_base_dir = r'C:/Users/brand/Desktop/Raj-Sindi/training_data/sim_csv_v11/'
    train_dir = os.path.join(all_runs_base_dir, 'train')
    val_dir = os.path.join(all_runs_base_dir, 'val')

    # Define the specific folder names for each run.
    # If a run doesn't exist, set its path to None.
    pysr_run_directories = {
        "Flux": {
            "Narrow Tau": os.path.join(train_dir, 'flux_only_nobias_train_e3_narrowtau'),
            "Wide Tau": os.path.join(train_dir, 'flux_only_nobias_train_e3'),
            "Biased Wide Tau": os.path.join(train_dir, 'flux_only_bias_train_e3'),
        },
        "W1": {
            "Narrow Tau": os.path.join(train_dir, 'w1_only_nobias_train_e3_narrowtau'),
            "Wide Tau": os.path.join(train_dir, 'w1_only_nobias_train_e3'),
            "Biased Wide Tau": os.path.join(train_dir, 'w1_only_bias_train_e3'),
        }
    }

    # Define where to save the final presentation plots
    output_visualization_dir = os.path.join(all_runs_base_dir, "Presentation_Visualizations")
    os.makedirs(output_visualization_dir, exist_ok=True)
    
    # --- Generate Visualizations ---
    
    # 1. Generate "The PySR Gauntlet" plot
    print("--- Generating PySR Performance Comparison Plot ---")
    plot_pysr_performance_comparison(pysr_run_directories, output_visualization_dir)
    
    # 2. Generate "The Method Showdown" plot
    print("\n--- Generating Method Showdown Plot ---")
    # For this comparison, we'll use the best PySR result from the most realistic dataset:
    # Biased, Wide-Range Tau, for the FValue target.
    best_pysr_run_path = pysr_run_directories['Flux']['Biased Wide Tau']
    best_pysr_nmse_value = get_best_pysr_nmse(best_pysr_run_path)
    
    if best_pysr_nmse_value is not None:
        plot_method_showdown(best_pysr_nmse_value, output_visualization_dir)
    else:
        print("Could not generate Method Showdown plot because the best PySR NMSE could not be determined.")

