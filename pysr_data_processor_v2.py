import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.utils import resample
from calculate_errors import *
import sympy as sp
import scipy.io
from collections import Counter, defaultdict
from wordcloud import WordCloud
from matplotlib.patches import ConnectionPatch
import re

# --- Matplotlib Settings ---
# Using the settings you provided for consistency.
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 24,
    'axes.labelsize': 28,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 32,
})
# --- End Settings ---

def find_and_load_model_data(base_dir):
    """
    Scours subdirectories of a base directory to find and load PySR model data.

    This function looks for 'model_equations.csv' and 'metrics_summary.csv',
    merges them, and compiles the data from all found runs into a single DataFrame.

    Args:
        base_dir (str): The path to the top-level directory containing the PySR run folders.

    Returns:
        pd.DataFrame: A unified DataFrame containing complexity, performance metrics,
                      and equations from all valid runs, or an empty DataFrame if no
                      data is found.
    """
    all_model_data = []
    print(f"🔍 Scouring for model data in: {base_dir}")

    for root, _, files in os.walk(base_dir):
        if "model_equations.csv" in files and "metrics_summary.csv" in files:
            try:
                eq_path = os.path.join(root, "model_equations.csv")
                metrics_path = os.path.join(root, "metrics_summary.csv")

                eq_df = pd.read_csv(eq_path)
                metrics_df = pd.read_csv(metrics_path)

                # Ensure dataframes can be merged
                if len(eq_df) != len(metrics_df):
                    print(f"⚠️ Warning: Row count mismatch in {root}. Skipping directory.")
                    continue

                # Combine the relevant columns with proper numeric conversion
                combined_df = pd.DataFrame({
                    'complexity': pd.to_numeric(eq_df.get('complexity'), errors='coerce'),
                    'equation': eq_df.get('equation'),
                    'sympy_format': eq_df.get('sympy_format'),
                    'NMSE_Train': pd.to_numeric(metrics_df.get('NMSE_Train'), errors='coerce'),
                    'NMSE_Test': pd.to_numeric(metrics_df.get('NMSE_Test'), errors='coerce'),
                    'NMSE_Validation': pd.to_numeric(metrics_df.get('NMSE_Validation'), errors='coerce'),
                    'run_id': os.path.basename(root) # Add a column to identify the run
                })
                
                all_model_data.append(combined_df)

            except Exception as e:
                print(f"❌ Error processing directory {root}: {e}")
                continue

    if not all_model_data:
        print("No valid model data was found.")
        return pd.DataFrame()

    # Concatenate all dataframes into one
    final_df = pd.concat(all_model_data, ignore_index=True)
    # Drop any rows where complexity could not be parsed OR where all NMSE values are NaN
    final_df.dropna(subset=['complexity'], inplace=True)
    
    # Also drop rows where ALL NMSE columns are NaN (optional, but helps with data quality)
    final_df = final_df.dropna(subset=['NMSE_Train', 'NMSE_Test', 'NMSE_Validation'], how='all')
    
    print(f"✅ Successfully loaded and combined data from {len(all_model_data)} runs.")
    return final_df

def analyze_best_equation_terms_by_dataset(base_directory, save_dir=None):
    # Variable translation function
    def translate_var(var):
        mapping = {
            "x0": "gamma",
            "x1": "lambda",
            "x2": "delta",
            "x3": "epsilon",
            "x4": "N1",
            "x5": "N2"
        }
        return mapping.get(var, var)

    """
    Scans all subdirectories for 'best.txt', extracts the best equation per dataset,
    analyzes normalized variable/operator frequency among best equations, and saves a table of best equations.
    """
    import re
    best_equations = []
    term_counter = Counter()
    op_counter = Counter()
    dataset_names = []

    # Scan for best.txt files
    for root, _, files in os.walk(base_directory):
        if "best.txt" in files:
            best_path = os.path.join(root, "best.txt")
            dataset_name = os.path.basename(root)
            with open(best_path, "r") as f:
                lines = f.readlines()
            # Find the line with the best equation
            eq_line = next((l for l in lines if l.startswith("Selected Best Function:")), None)
            if eq_line:
                # Format: Selected Best Function: 10, -x0/exp(x0) + exp(x5)*(-0.3245)
                match = re.match(r"Selected Best Function:\s*\d+,\s*(.*)", eq_line)
                if match:
                    equation = match.group(1).strip()
                    best_equations.append({"dataset": dataset_name, "equation": equation})
                    dataset_names.append(dataset_name)
                    # Parse equation for terms/operators
                    ops, terms, _ = extract_terms_and_ops(equation)
                    for t in terms:
                        term_counter[t] += 1
                    for o in ops:
                        op_counter[o] += 1

    # Normalize frequencies
    total_eqs = len(best_equations)
    norm_term_freq = {k: v/total_eqs for k, v in term_counter.items()}
    norm_op_freq = {k: v/total_eqs for k, v in op_counter.items()}

    # Only plot variable frequency with labels, ordered x0-x5, translated names, and consistent color
    ordered_vars = ["x0", "x1", "x2", "x3", "x4", "x5"]
    ordered_freqs = [norm_term_freq.get(v, 0) for v in ordered_vars]
    translated_labels = [translate_var(v) for v in ordered_vars]
    bar_color = "#84A8C5"  # Consistent color as before
    plt.figure(figsize=(8,4))
    bars = plt.bar(translated_labels, ordered_freqs, color=bar_color)
    plt.title("Average Term Frequency Across Output Models")#,fontsize=20)
    plt.ylabel("Frequency",fontsize=16)
    #plt.xlabel("")
    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    plt.tight_layout()
    # Add value labels on top or inside bars
    for bar, freq in zip(bars, ordered_freqs):
        xpos = bar.get_x() + bar.get_width()/2
        if freq < 0.15:
            plt.text(xpos, freq + 0.01, f"{freq:.2f}", ha='center', va='bottom', fontsize=16)
        else:
            plt.text(xpos, freq/2, f"{freq:.2f}", ha='center', va='center', color='white', fontsize=16, fontweight='bold')
    if save_dir:
        plt.savefig(os.path.join(save_dir, "best_equation_variable_frequency.png"), dpi=300)
    plt.show()

    # Save table of best equations
    if save_dir and best_equations:
        import pandas as pd
        df = pd.DataFrame(best_equations)
        table_path = os.path.join(save_dir, "best_equations_by_dataset.csv")
        df.to_csv(table_path, index=False)
        print(f"Saved best equations table to {table_path}")

    return best_equations, norm_term_freq, norm_op_freq

def load_scaling_params(model_path):
    """Load scaling parameters from scaling_params.mat in the model directory."""
    mat_path = os.path.join(os.path.dirname(model_path), "scaling_params.mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Scaling file not found: {mat_path}")
    
    mat = scipy.io.loadmat(mat_path, squeeze_me=True)
    scaling = {
        "feature_names": [str(f) for f in mat["feature_names"].tolist()],
        "feature_means": np.array(mat["feature_means"], dtype=float),
        "feature_stds": np.array(mat["feature_stds"], dtype=float),
        "target_mean": float(mat["target_mean"]),
        "target_std": float(mat["target_std"]),
    }
    return scaling

def apply_scaling(df, target, scaling, feature_cols):
    """Apply saved scaling to dataset (features and target)."""
    # 🎯 Use the provided feature_cols list instead of inferring them
    X = df[feature_cols].values
    y = df[target].values

    # Apply scaling
    X_scaled = (X - scaling["feature_means"]) / scaling["feature_stds"]
    y_scaled = (y - scaling["target_mean"]) / scaling["target_std"]

    return X_scaled, y_scaled, feature_cols

def inverse_scale_predictions(y_scaled_pred, scaling):
    """Inverse transform predictions to original target scale."""
    return y_scaled_pred * scaling["target_std"] + scaling["target_mean"]

def nmse_bootstrap_pysr_sympy(equations, X, y, n_bootstrap=100):
    """
    Compute NMSE for equations from PySR using sympy parsing + lambdify.
    
    Parameters
    ----------
    equations : list of str
        List of equation strings (from CSV: sympy_format column).
    X : np.ndarray
        Input features, shape (n_samples, n_features).
    y : np.ndarray
        Ground truth target values, shape (n_samples,).
    n_bootstrap : int
        Number of bootstrap samples.
    
    Returns
    -------
    list of float
        NMSE values for each equation.
    """
    n_samples, n_features = X.shape
    nmse_list = []

    # Define sympy symbols x0, x1, ..., xN
    symbols = sp.symbols([f"x{i}" for i in range(n_features)])

    for eq in equations:
        try:
            # Parse equation string into sympy expression
            expr = sp.sympify(eq)

            # Turn into a numpy-compatible function
            func = sp.lambdify(symbols, expr, modules="numpy")

            # Evaluate predictions
            y_pred = func(*[X[:, i] for i in range(n_features)])
            y_pred = np.array(y_pred, dtype=float).flatten()

            # Bootstrap NMSE
            nmse_bootstrap = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(n_samples, size=n_samples, replace=True)
                mse = np.mean((y[idx] - y_pred[idx])**2)
                var = np.var(y[idx])
                nmse_bootstrap.append(mse / var if var != 0 else np.inf)

            nmse_list.append(np.mean(nmse_bootstrap))

        except Exception as e:
            print(f"Skipping equation {eq}: {e}")
            nmse_list.append(np.nan)

    return nmse_list

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


def plot_nmse_vs_complexity_from_subdirs(base_dir, output_dir=None):
    """
    For each subdirectory in base_dir, load 'model_equations.csv' and 'metrics_summary.csv',
    extract complexity and NMSE (9th column), and plot NMSE vs complexity for each run.
    Also, mark the "best" equation (from best.txt) as a dot with the same color as the line but brighter.

    Args:
        base_dir (str): Directory containing subdirectories for each run.
        output_dir (str): Directory to save the figure. If None, only shows the plot.
    """
    import matplotlib.colors as mcolors

    plt.figure(figsize=(12, 8))
    found_any = False
    line_colors = {}  # Track colors for each subdirectory

    for subdir in os.listdir(base_dir):
        sub_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(sub_path):
            continue
        eq_path = os.path.join(sub_path, "model_equations.csv")
        metrics_path = os.path.join(sub_path, "metrics_summary.csv")
        best_path = os.path.join(sub_path, "best.txt")
        if os.path.exists(eq_path) and os.path.exists(metrics_path):
            try:
                eq_df = pd.read_csv(eq_path)
                metrics_df = pd.read_csv(metrics_path)
                # Complexity column (assume named 'complexity')
                if 'complexity' in eq_df.columns:
                    complexity = eq_df['complexity'].values
                else:
                    complexity = eq_df.iloc[:, 0].values
                # NMSE column (9th column, index 8)
                nmse = metrics_df.iloc[:, 8].values
                # If lengths mismatch, skip
                if len(complexity) != len(nmse):
                    print(f"Warning: Row count mismatch in {sub_path}")
                    continue
                label = subdir
                
                # Plot the line and capture the color
                line = plt.plot(complexity, nmse, marker='o', linestyle='-', label=label)
                line_color = line[0].get_color()  # Get the actual color used by matplotlib
                line_colors[subdir] = line_color
                found_any = True

                # --- Mark the best equation with a brighter version of the line color ---
                if os.path.exists(best_path):
                    with open(best_path, "r") as f:
                        lines = f.readlines()
                    eq_line = next((l for l in lines if l.startswith("Selected Best Function:")), None)
                    if eq_line:
                        match = re.match(r"Selected Best Function:\s*(\d+),", eq_line)
                        if match:
                            best_idx = int(match.group(1))
                            # Find the index of the equation with this complexity
                            if best_idx in complexity:
                                idxs = np.where(complexity == best_idx)[0]
                                # If multiple, pick the first
                                idx = idxs[0]
                                
                                # Create a brighter version of the line color
                                try:
                                    # Convert color to RGB if it's not already
                                    rgb = mcolors.to_rgb(line_color)
                                    # Brighten the color by moving it closer to white
                                    # You can adjust the factor (0.3) to make it more or less bright
                                    bright_rgb = tuple(min(1.0, c + (1.0 - c) * 0.4) for c in rgb)
                                    bright_color = bright_rgb
                                except:
                                    # Fallback to original color if conversion fails
                                    bright_color = line_color
                                
                                plt.scatter(complexity[idx], nmse[idx], color=bright_color, s=150, 
                                          zorder=10, edgecolors='white', linewidth=2,
                                          label=f"{label} Best" if idx == 0 else "")  # Only label the first one
                            else:
                                # If not found, skip marking
                                print(f"Best equation complexity {best_idx} not found in {sub_path}")
            except Exception as e:
                print(f"Error processing {sub_path}: {e}")
                continue

    plt.xlabel("Complexity")
    plt.ylabel("NMSE")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("NMSE vs Complexity Across Runs")
    #if found_any:
        #plt.legend()   
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "nmse_vs_complexity_comparison.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    plt.show()


# Function that scours each folder and subfolder for a .csv called "metrics_summary" then finds each NMSE value and associated complexity, and adds them to a list.
# Then, it'll take the average of NMSE as complexity increases and graph it:
def analyze_nmse_by_complexity_subplots(directories, save_path=None, plot_title=None):
    """
    Generate subplots for NMSE vs Complexity for each directory in 'directories'.
    Pairs flux and w1 datasets by row with readable titles.
    Args:
        directories (list): List of directory paths.
        save_path (str): Path to save the figure.
        plot_title (str): Title for the whole figure.
    Returns:
        dict: Mapping of label -> results DataFrame.
    """
    
    # Function to get readable titles
    def get_readable_title(dir_path):
        dir_name = os.path.basename(dir_path)
        title_mapping = {
            "flux_only_bias_train_e3": "Bias, Flux",
            "flux_only_nobias_train_e3": "Wide Tau, Flux", 
            "flux_only_nobias_train_e3_narrowtau": "Narrow Tau, Flux",
            "w1_only_bias_train_e3": "Bias, W1",
            "w1_only_nobias_train_e3": "Wide Tau, W1",
            "w1_only_nobias_train_e3_narrowtau": "Narrow Tau, W1"
        }
        return title_mapping.get(dir_name, dir_name)
    
    # Create directory pairs based on the pattern
    def pair_directories(directories):
        dir_dict = {}
        for dir_path in directories:
            dir_name = os.path.basename(dir_path)
            if "flux_only_bias" in dir_name:
                key = "bias"
            elif "flux_only_nobias_train_e3_narrowtau" in dir_name:
                key = "narrow_tau"
            elif "flux_only_nobias" in dir_name:
                key = "wide_tau"
            elif "w1_only_bias" in dir_name:
                key = "bias"
            elif "w1_only_nobias_train_e3_narrowtau" in dir_name:
                key = "narrow_tau"
            elif "w1_only_nobias" in dir_name:
                key = "wide_tau"
            else:
                continue
                
            if key not in dir_dict:
                dir_dict[key] = {"flux": None, "w1": None}
            
            if "flux_only" in dir_name:
                dir_dict[key]["flux"] = dir_path
            elif "w1_only" in dir_name:
                dir_dict[key]["w1"] = dir_path
        
        # Convert to ordered pairs
        order = ["bias", "wide_tau", "narrow_tau"]
        pairs = []
        for key in order:
            if key in dir_dict:
                flux_dir = dir_dict[key]["flux"]
                w1_dir = dir_dict[key]["w1"]
                if flux_dir or w1_dir:  # At least one exists
                    pairs.append((flux_dir, w1_dir))
        
        return pairs
    
    # Get paired directories
    directory_pairs = pair_directories(directories)
    n_rows = len(directory_pairs)
    n_cols = 2  # Always 2 columns (flux and w1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8 * n_rows), sharey=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    results_dfs = {}

    for row, (flux_dir, w1_dir) in enumerate(directory_pairs):
        for col, base_dir in enumerate([flux_dir, w1_dir]):
            if base_dir is None:
                # Hide subplot if directory doesn't exist
                axes[row, col].set_visible(False)
                continue
                
            ax = axes[row, col]
            label = os.path.basename(base_dir)

            # Use the unified loader to get all model data for this directory
            results_df = find_and_load_model_data(base_dir)
            results_df = results_df.rename(columns={
                'complexity': 'Complexity',
                'NMSE_Train': 'NMSE_Train',
                'NMSE_Test': 'NMSE_Test',
                'NMSE_Validation': 'NMSE_Validation'
            })

            results_dfs[label] = results_df

            if len(results_df) == 0:
                ax.set_title(f"{get_readable_title(base_dir)}\n(No valid data)", fontsize=32)
                continue

            # Only select numeric columns for groupby operation
            numeric_cols = ['Complexity', 'NMSE_Train', 'NMSE_Test', 'NMSE_Validation']
            numeric_df = results_df[numeric_cols].copy()
            
            # Additional cleaning - drop any remaining NaN values
            numeric_df = numeric_df.dropna()
            
            if len(numeric_df) == 0:
                ax.set_title(f"{get_readable_title(base_dir)}\n(No valid numeric data)", fontsize=32)
                continue
            
            grouped = numeric_df.groupby('Complexity').mean().reset_index()
            
            # Plot with improved styling
            ax.plot(grouped['Complexity'], grouped['NMSE_Train'], 
                   marker='o', label='Training', alpha=0.8, linewidth=4, 
                   markersize=10, color='#95BAA6')
            ax.plot(grouped['Complexity'], grouped['NMSE_Test'], 
                   marker='s', label='Testing', alpha=0.8, linewidth=4, 
                   markersize=10, color='#EFE868')
            ax.plot(grouped['Complexity'], grouped['NMSE_Validation'], 
                   marker='^', label='Validation', alpha=0.8, linewidth=4, 
                   markersize=10, color='#84A8C5')
            
            ax.set_yscale('log')
            ax.set_title(get_readable_title(base_dir), fontsize=32)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            
            # Control tick labels based on position
            if row == n_rows - 1:  # Bottom row - show x-tick labels
                ax.tick_params(axis='x', labelsize=24, labelbottom=True)
            else:  # Other rows - hide x-tick labels
                ax.tick_params(axis='x', labelsize=24, labelbottom=False)
            
            ax.tick_params(axis='y', labelsize=24)

            # Only show legend on the top-left subplot with larger font
            if row == 0 and col == 0:
                legend = ax.legend(fontsize=24, frameon=True, fancybox=True, shadow=True)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
            
            # Only show y-label on the middle left subplot
            if col == 0 and row == len(directory_pairs) // 2:
                ax.set_ylabel('Average NMSE (Log Scale)', fontsize=28)
            
            # Only show x-label on the bottom row (both subplots)
            if row == n_rows - 1:
                ax.set_xlabel(' ', fontsize=26)

    fig.suptitle(plot_title if plot_title else "Average NMSE by Model Complexity")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved NMSE by complexity subplots to {save_path}")
    plt.show()
    return results_dfs


def analyze_nmse_by_complexity(base_dir, save_path=None, plot_title=None):
    nmse_train = []
    nmse_test = []
    nmse_val = []
    complexities = []

    for root, dirs, files in os.walk(base_dir):
        if "metrics_summary.csv" in files and "model_equations.csv" in files:
            metrics_path = os.path.join(root, "metrics_summary.csv")
            equations_path = os.path.join(root, "model_equations.csv")
            metrics_df = pd.read_csv(metrics_path)
            
            # Read equations CSV with header
            equations_df = pd.read_csv(equations_path)
            
            # Extract complexity values and convert to numeric
            if len(equations_df) == len(metrics_df):
                complexity_list = pd.to_numeric(equations_df['complexity'], errors='coerce').values
            else:
                print(f"Warning: Row count mismatch in {root}")
                complexity_list = [np.nan] * len(metrics_df)

            # Extract NMSE columns
            train_list = metrics_df['NMSE_Train'].values if 'NMSE_Train' in metrics_df.columns else [np.nan] * len(metrics_df)
            test_list = metrics_df['NMSE_Test'].values if 'NMSE_Test' in metrics_df.columns else [np.nan] * len(metrics_df)
            val_list = metrics_df['NMSE_Validation'].values if 'NMSE_Validation' in metrics_df.columns else [np.nan] * len(metrics_df)

            nmse_train.extend(train_list)
            nmse_test.extend(test_list)
            nmse_val.extend(val_list)
            complexities.extend(complexity_list)

    # Create DataFrame
    results_df = pd.DataFrame({
        'Complexity': complexities,
        'NMSE_Train': nmse_train,
        'NMSE_Test': nmse_test,
        'NMSE_Validation': nmse_val
    })

    # Clean the data
    results_df = results_df.dropna(subset=['Complexity'])
    
    if len(results_df) == 0:
        print("ERROR: No valid data remaining after cleaning!")
        return

    # Group by complexity and average
    grouped = results_df.groupby('Complexity').mean().reset_index()

    # Create main plot with inset
    fig, ax_main = plt.subplots(figsize=(12, 8))
    
    # Plot main data
    ax_main.plot(grouped['Complexity'], grouped['NMSE_Train'], marker='o', label='NMSE Train', alpha=0.8, linewidth=2, color='#95BAA6')
    ax_main.plot(grouped['Complexity'], grouped['NMSE_Test'], marker='s', label='NMSE Test', alpha=0.8, linewidth=2, color='#B5E881')
    ax_main.plot(grouped['Complexity'], grouped['NMSE_Validation'], marker='^', label='NMSE Validation', alpha=0.8, linewidth=2, color='#84A8C5')
    
    #ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_xlabel('Model Complexity')
    ax_main.set_ylabel('Average NMSE (Log Scale)')
    ax_main.set_title('Average NMSE by Model Complexity' if plot_title is None else plot_title)
    ax_main.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_main.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved NMSE by complexity plot with inset to {save_path}")
    plt.show()
    
    return results_df


def analyze_nmse_by_complexity_with_inset(base_dir, save_path=None, plot_title=None):
    nmse_train = []
    nmse_test = []
    nmse_val = []
    complexities = []

    for root, dirs, files in os.walk(base_dir):
        if "metrics_summary.csv" in files and "model_equations.csv" in files:
            metrics_path = os.path.join(root, "metrics_summary.csv")
            equations_path = os.path.join(root, "model_equations.csv")
            metrics_df = pd.read_csv(metrics_path)
            
            # Read equations CSV with header
            equations_df = pd.read_csv(equations_path)
            
            # Extract complexity values and convert to numeric
            if len(equations_df) == len(metrics_df):
                complexity_list = pd.to_numeric(equations_df['complexity'], errors='coerce').values
            else:
                print(f"Warning: Row count mismatch in {root}")
                complexity_list = [np.nan] * len(metrics_df)

            # Extract NMSE columns
            train_list = metrics_df['NMSE_Train'].values if 'NMSE_Train' in metrics_df.columns else [np.nan] * len(metrics_df)
            test_list = metrics_df['NMSE_Test'].values if 'NMSE_Test' in metrics_df.columns else [np.nan] * len(metrics_df)
            val_list = metrics_df['NMSE_Validation'].values if 'NMSE_Validation' in metrics_df.columns else [np.nan] * len(metrics_df)

            nmse_train.extend(train_list)
            nmse_test.extend(test_list)
            nmse_val.extend(val_list)
            complexities.extend(complexity_list)

    # Create DataFrame
    results_df = pd.DataFrame({
        'Complexity': complexities,
        'NMSE_Train': nmse_train,
        'NMSE_Test': nmse_test,
        'NMSE_Validation': nmse_val
    })

    # Clean the data
    results_df = results_df.dropna(subset=['Complexity'])
    
    if len(results_df) == 0:
        print("ERROR: No valid data remaining after cleaning!")
        return

    # Group by complexity and average
    grouped = results_df.groupby('Complexity').mean().reset_index()

    # Create main plot with inset
    fig, ax_main = plt.subplots(figsize=(12, 8))
    
    # Plot main data
    ax_main.plot(grouped['Complexity'], grouped['NMSE_Train'], marker='o', label='NMSE Train', alpha=0.8, linewidth=2,color='#95BAA6')
    ax_main.plot(grouped['Complexity'], grouped['NMSE_Test'], marker='s', label='NMSE Test', alpha=0.8, linewidth=2,color="#EFE868")
    ax_main.plot(grouped['Complexity'], grouped['NMSE_Validation'], marker='^', label='NMSE Validation', alpha=0.8, linewidth=2,color='#84A8C5')

    #ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_xlabel('Model Complexity')
    ax_main.set_ylabel('Average NMSE (Log Scale)')
    ax_main.set_title('Average NMSE by Model Complexity' if plot_title is None else plot_title)
    ax_main.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax_main.legend()

    # Create inset plot for bias-variance tradeoff region
    # Position: [left, bottom, width, height] in figure coordinates
    # Centered and smaller to avoid legend
    inset_ax = fig.add_axes([0.16, 0.45, 0.38, 0.35])  # More centered position
    
    # Filter data for inset - focus on narrow range around the minimum
    # Find approximate minimum of test/validation error for better range
    test_min_idx = grouped['NMSE_Test'].idxmin()
    val_min_idx = grouped['NMSE_Validation'].idxmin()
    center_complexity = grouped.loc[min(test_min_idx, val_min_idx), 'Complexity']
    
    # Show 2-3 points to the left and right of the minimum
    inset_min_c = max(1, center_complexity - 6)
    inset_max_c = center_complexity - 2
    inset_data = grouped[(grouped['Complexity'] >= inset_min_c) & (grouped['Complexity'] <= inset_max_c)]
    
    if len(inset_data) > 0:
        inset_ax.plot(inset_data['Complexity'], inset_data['NMSE_Train'], 
                     marker='o', alpha=0.9, linewidth=2, markersize=6,color='#95BAA6')
        inset_ax.plot(inset_data['Complexity'], inset_data['NMSE_Test'], 
                     marker='s', alpha=0.9, linewidth=2, markersize=6,color="#EFE868")
        inset_ax.plot(inset_data['Complexity'], inset_data['NMSE_Validation'], 
                     marker='^', alpha=0.9, linewidth=2, markersize=6,color='#84A8C5')

        # Clean up inset appearance - no legend, minimal labels
        #inset_ax.set_xlabel('Complexity', fontsize=9)
        #inset_ax.set_ylabel('NMSE', fontsize=9)
        #inset_ax.set_title('Bias-Variance Tradeoff', fontsize=10)
        inset_ax.grid(True, alpha=0.3)
        inset_ax.tick_params(labelsize=16)
        
        # Set reasonable y-limits for the inset to focus on the tradeoff
        #y_min = min(inset_data[['NMSE_Train', 'NMSE_Test', 'NMSE_Validation']].min())
        #y_max = max(inset_data[['NMSE_Train', 'NMSE_Test', 'NMSE_Validation']].max())
        y_min = min(inset_data[['NMSE_Train', 'NMSE_Test']].min())
        y_max = max(inset_data[['NMSE_Train', 'NMSE_Test']].max())
        inset_ax.set_ylim(y_min * 0.95, y_max * 1.05)
        inset_ax.set_xlim(inset_min_c - 0.2, inset_max_c + 0.2)
        #inset_ax.set_yscale('log')

        # Add a rectangle on the main plot to show the inset region
        from matplotlib.patches import Rectangle
        rect = Rectangle((inset_min_c, y_min * 0.95), 
                        inset_max_c - inset_min_c, 
                        (y_max * 1.05) - (y_min * 0.95), 
                        linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--', alpha=0.7)
        ax_main.add_patch(rect)
        
        # Add an arrow or connection line from rectangle to inset

        # Connection from rectangle to inset
        con = ConnectionPatch(xyA=(inset_max_c, y_max * 1.05), coordsA=ax_main.transData,
                            xyB=(0, 1), coordsB=inset_ax.transAxes,
                            arrowstyle='->', color='red', alpha=0.7, linewidth=1)
        ax_main.add_patch(con)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved NMSE by complexity plot with inset to {save_path}")
    plt.show()
    
    return results_df

def analyze_nmse_by_complexity_with_inset2(base_dir, save_path=None, plot_title=None):
    nmse_train = []
    nmse_test = []
    nmse_val = []
    complexities = []

    for root, dirs, files in os.walk(base_dir):
        if "metrics_summary.csv" in files and "model_equations.csv" in files:
            metrics_path = os.path.join(root, "metrics_summary.csv")
            equations_path = os.path.join(root, "model_equations.csv")
            metrics_df = pd.read_csv(metrics_path)
            
            # Read equations CSV with header
            equations_df = pd.read_csv(equations_path)
            
            # Extract complexity values and convert to numeric
            if len(equations_df) == len(metrics_df):
                complexity_list = pd.to_numeric(equations_df['complexity'], errors='coerce').values
            else:
                print(f"Warning: Row count mismatch in {root}")
                complexity_list = [np.nan] * len(metrics_df)

            # Extract NMSE columns
            train_list = metrics_df['NMSE_Train'].values if 'NMSE_Train' in metrics_df.columns else [np.nan] * len(metrics_df)
            test_list = metrics_df['NMSE_Test'].values if 'NMSE_Test' in metrics_df.columns else [np.nan] * len(metrics_df)
            val_list = metrics_df['NMSE_Validation'].values if 'NMSE_Validation' in metrics_df.columns else [np.nan] * len(metrics_df)

            nmse_train.extend(train_list)
            nmse_test.extend(test_list)
            nmse_val.extend(val_list)
            complexities.extend(complexity_list)

    # Create DataFrame
    results_df = pd.DataFrame({
        'Complexity': complexities,
        'NMSE_Train': nmse_train,
        'NMSE_Test': nmse_test,
        'NMSE_Validation': nmse_val
    })

    # Clean the data
    results_df = results_df.dropna(subset=['Complexity'])
    
    if len(results_df) == 0:
        print("ERROR: No valid data remaining after cleaning!")
        return

    # Group by complexity and average
    grouped = results_df.groupby('Complexity').mean().reset_index()

    # Create main plot with inset
    fig, ax_main = plt.subplots(figsize=(12, 8))
    
    # Plot main data with improved styling
    line1 = ax_main.plot(grouped['Complexity'], grouped['NMSE_Train'], 
                        marker='o', label='Training', alpha=0.8, linewidth=3, 
                        markersize=8, color='#95BAA6')
    line2 = ax_main.plot(grouped['Complexity'], grouped['NMSE_Test'], 
                        marker='s', label='Testing', alpha=0.8, linewidth=3, 
                        markersize=8, color='#EFE868')
    line3 = ax_main.plot(grouped['Complexity'], grouped['NMSE_Validation'], 
                        marker='^', label='Validation', alpha=0.8, linewidth=3, 
                        markersize=8, color='#84A8C5')

    ax_main.set_yscale('log')
    ax_main.set_xlabel('Model Complexity', fontsize=18)
    ax_main.set_ylabel('Average NMSE (Log Scale)', fontsize=18)
    ax_main.set_title('Average NMSE by Model Complexity' if plot_title is None else plot_title, fontsize=20)
    ax_main.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Improve legend
    legend = ax_main.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Improve tick labels
    ax_main.tick_params(axis='x', labelsize=14)
    ax_main.tick_params(axis='y', labelsize=14)

    # Create inset plot for bias-variance tradeoff region
    inset_ax = fig.add_axes([0.16, 0.45, 0.38, 0.35])
    
    # Filter data for inset - focus on narrow range around the minimum
    test_min_idx = grouped['NMSE_Test'].idxmin()
    val_min_idx = grouped['NMSE_Validation'].idxmin()
    center_complexity = grouped.loc[min(test_min_idx, val_min_idx), 'Complexity']
    
    # Show 2-3 points to the left and right of the minimum
    inset_min_c = max(1, center_complexity - 6)
    inset_max_c = center_complexity - 2
    inset_data = grouped[(grouped['Complexity'] >= inset_min_c) & (grouped['Complexity'] <= inset_max_c)]
    
    if len(inset_data) > 0:
        inset_ax.plot(inset_data['Complexity'], inset_data['NMSE_Train'], 
                     marker='o', alpha=0.9, linewidth=3, markersize=8, color='#95BAA6')
        inset_ax.plot(inset_data['Complexity'], inset_data['NMSE_Test'], 
                     marker='s', alpha=0.9, linewidth=3, markersize=8, color='#EFE868')
        inset_ax.plot(inset_data['Complexity'], inset_data['NMSE_Validation'], 
                     marker='^', alpha=0.9, linewidth=3, markersize=8, color='#84A8C5')

        inset_ax.grid(True, alpha=0.3)
        inset_ax.tick_params(labelsize=12)
        
        # Set reasonable y-limits for the inset to focus on the tradeoff
        y_min = min(inset_data[['NMSE_Train', 'NMSE_Test']].min())
        y_max = max(inset_data[['NMSE_Train', 'NMSE_Test']].max())
        inset_ax.set_ylim(y_min * 0.95, y_max * 1.05)
        inset_ax.set_xlim(inset_min_c - 0.2, inset_max_c + 0.2)

        # Add a rectangle on the main plot to show the inset region
        from matplotlib.patches import Rectangle
        rect = Rectangle((inset_min_c, y_min * 0.95), 
                        inset_max_c - inset_min_c, 
                        (y_max * 1.05) - (y_min * 0.95), 
                        linewidth=2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.8)
        ax_main.add_patch(rect)
        
        # Connection from rectangle to inset
        con = ConnectionPatch(xyA=(inset_max_c, y_max * 1.05), coordsA=ax_main.transData,
                            xyB=(0, 1), coordsB=inset_ax.transAxes,
                            arrowstyle='->', color='red', alpha=0.8, linewidth=2)
        ax_main.add_patch(con)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved NMSE by complexity plot with inset to {save_path}")
    plt.show()
    
    return results_df

def create_pareto_front_plot(base_dir, save_path=None):
    """
    Create a Pareto front plot showing complexity vs performance for all discovered models
    """
    all_models = []
    
    # Collect all individual models (not averaged)
    for root, dirs, files in os.walk(base_dir):
        if "metrics_summary.csv" in files and "model_equations.csv" in files:
            metrics_path = os.path.join(root, "metrics_summary.csv")
            equations_path = os.path.join(root, "model_equations.csv")
            
            try:
                metrics_df = pd.read_csv(metrics_path)
                equations_df = pd.read_csv(equations_path)
                
                # Extract run identifier from path (optional, for coloring)
                run_id = root.split(os.sep)[-1] if root.split(os.sep)[-1] else "unknown"
                
                # Match models from both files
                if len(equations_df) == len(metrics_df):
                    complexities = pd.to_numeric(equations_df['complexity'], errors='coerce')
                    
                    # Choose which metric to use (validation NMSE is often best for generalization)
                    if 'NMSE_Validation' in metrics_df.columns:
                        performances = metrics_df['NMSE_Validation']
                        metric_name = 'NMSE_Validation'
                    elif 'NMSE_Test' in metrics_df.columns:
                        performances = metrics_df['NMSE_Test']
                        metric_name = 'NMSE_Test'
                    else:
                        performances = metrics_df['NMSE_Train']
                        metric_name = 'NMSE_Train'
                    
                    # Add each model as individual data point
                    for i, (complexity, performance) in enumerate(zip(complexities, performances)):
                        if not (pd.isna(complexity) or pd.isna(performance)):
                            all_models.append({
                                'complexity': complexity,
                                'performance': performance,
                                'run_id': run_id,
                                'model_idx': i
                            })
                            
            except Exception as e:
                print(f"Error processing {root}: {e}")
                continue
    
    if not all_models:
        print("No valid models found!")
        return
    
    # Convert to DataFrame
    models_df = pd.DataFrame(all_models)
    print(f"Total models collected: {len(models_df)}")
    print(f"Complexity range: {models_df['complexity'].min():.1f} to {models_df['complexity'].max():.1f}")
    print(f"Performance range: {models_df['performance'].min():.6f} to {models_df['performance'].max():.6f}")
    
    # Find Pareto front
    def find_pareto_front(df):
        """
        Find Pareto optimal points (minimize both complexity and performance)
        """
        pareto_points = []
        
        # Sort by complexity for easier processing
        sorted_df = df.sort_values('complexity')
        
        for i, row in sorted_df.iterrows():
            is_pareto = True
            
            # Check if any other point dominates this one
            # (has lower complexity AND lower performance)
            for j, other_row in sorted_df.iterrows():
                if (other_row['complexity'] <= row['complexity'] and 
                    other_row['performance'] < row['performance']):
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_points.append(row)
        
        return pd.DataFrame(pareto_points)
    
    pareto_df = find_pareto_front(models_df)
    pareto_df = pareto_df.sort_values('complexity')
    
    print(f"Pareto optimal models: {len(pareto_df)}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot all models as scatter points
    plt.scatter(models_df['complexity'], models_df['performance'], 
                alpha=0.4, s=30, c='lightblue', label=f'All Models (n={len(models_df)})')
    
    # Highlight Pareto front
    plt.scatter(pareto_df['complexity'], pareto_df['performance'], 
                alpha=0.8, s=60, c='red', label=f'Pareto Front (n={len(pareto_df)})', 
                edgecolors='darkred', linewidth=1)
    
    # Connect Pareto front points
    plt.plot(pareto_df['complexity'], pareto_df['performance'], 
             'r--', alpha=0.7, linewidth=2, label='Pareto Frontier')
    
    # Use log scale for better visualization
    plt.yscale('log')
    plt.xlabel('Model Complexity')
    #plt.ylabel(f'{metric_name} (Log Scale)')
    plt.ylabel('Validation NMSE (Log Scale)')

    plt.title('Pareto Front: Model Complexity vs Performance\n(All Discovered Models)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics as text
    best_simple = models_df[models_df['complexity'] <= 5]['performance'].min()
    best_complex = models_df[models_df['complexity'] >= 20]['performance'].min()
    
    plt.text(0.02, 0.98, 
             f'Best simple model (≤5): {best_simple:.4f}\n'
             f'Best complex model (≥20): {best_complex:.4f}\n'
             f'Improvement: {best_simple/best_complex:.2f}x',
             transform=plt.gca().transAxes, fontsize=18,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Pareto front plot to {save_path}")
    plt.show()
    
    return models_df, pareto_df

def evaluate_models_on_datasets_structured(output_dir, n_bootstrap=10):
    """
    Evaluate models on datasets in a structured 3x2xN bootstrap matrix format.
    
    - Runs separately for train and validation datasets
    - Uses evaluate_single_equation_scaled (scales features, inverse-transforms predictions)
    - Saves results in multiple CSV/NPY formats for reporting
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")

    # === Model equation paths ===
    model_equations_paths = [
        'C:/Users/brand/Desktop/School/Quals/data/train/flux_only_bias_train_e3/2025-08-20_09-46_172800_Info/model_equations.csv',
        'C:/Users/brand/Desktop/School/Quals/data/train/flux_only_nobias_train_e3/2025-08-20_09-46_172800_Info/model_equations.csv',
        'C:/Users/brand/Desktop/School/Quals/data/train/flux_only_nobias_train_e3_narrowtau/2025-08-24_11-06_172800_Info/model_equations.csv',
        'C:/Users/brand/Desktop/School/Quals/data/train/w1_only_bias_train_e3/2025-08-24_23-13_172800_Info/model_equations.csv',
        'C:/Users/brand/Desktop/School/Quals/data/train/w1_only_nobias_train_e3/2025-08-21_01-03_172800_Info/model_equations.csv',
        'C:/Users/brand/Desktop/School/Quals/data/train/w1_only_nobias_train_e3_narrowtau/2025-08-21_01-05_172800_Info/model_equations.csv'
    ]

    base_data_dir = r'C:\Users\brand\Desktop\Raj-Sindi\training_data\sim_csv_v11\e3_data'

    # Train datasets
    train_datasets = [
        {"name": "bias_train_e3", "csv_path": f"{base_data_dir}/bias_train_e3.csv",
         "fvalue_model": model_equations_paths[0], "w1_model": model_equations_paths[3]},
        {"name": "nobias_train_e3", "csv_path": f"{base_data_dir}/nobias_train_e3.csv",
         "fvalue_model": model_equations_paths[1], "w1_model": model_equations_paths[4]},
        {"name": "nobias_train_e3_narrowtau", "csv_path": f"{base_data_dir}/nobias_train_e3_narrowtau.csv",
         "fvalue_model": model_equations_paths[2], "w1_model": model_equations_paths[5]},
    ]

    # Validation datasets
    val_datasets = [
        {"name": "bias_val_e3", "csv_path": f"{base_data_dir}/bias_val_e3.csv",
         "fvalue_model": model_equations_paths[0], "w1_model": model_equations_paths[3]},
        {"name": "nobias_val_e3", "csv_path": f"{base_data_dir}/nobias_val_e3.csv",
         "fvalue_model": model_equations_paths[1], "w1_model": model_equations_paths[4]},
        {"name": "nobias_val_e3_narrowtau", "csv_path": f"{base_data_dir}/nobias_val_e3_narrowtau.csv",
         "fvalue_model": model_equations_paths[2], "w1_model": model_equations_paths[5]}
    ]

    targets = ["FValue", "W1"]
    feature_cols = ['gamma1', 'lambda1', 'delta', 'epsilon', 'NRow', 'NCol']

    # === helper function for one pass ===
    def evaluate_and_save(datasets, label):
        results_matrix = np.full((3, 2, n_bootstrap), np.nan)
        results_metadata = []
        summary_stats = []

        print(f"\n=== Starting {label.upper()} evaluation ===")

        for dataset_idx, dataset_info in enumerate(datasets):
            print(f"\n--- Dataset {dataset_info['name']} ---")
            df = pd.read_csv(dataset_info["csv_path"]).dropna()

            for target_idx, target in enumerate(targets):
                model_path = dataset_info["fvalue_model"] if target == "FValue" else dataset_info["w1_model"]

                try:
                    eq_df = pd.read_csv(model_path)
                    equation = eq_df["sympy_format"].iloc[-1]
                    complexity = eq_df["complexity"].iloc[-1]

                    scaling = load_scaling_params(model_path)
                    X_scaled, y_scaled, _ = apply_scaling(df, target, scaling, feature_cols=feature_cols)

                    print(f"  {target}: eq (complexity {complexity}) → {equation}")

                    bootstrap_nmse = []
                    for bootstrap_idx in range(n_bootstrap):
                        np.random.seed(42 + bootstrap_idx)
                        idx = np.random.choice(len(y_scaled), size=len(y_scaled), replace=True)
                        nmse_val = evaluate_single_equation_scaled(
                            equation, X_scaled[idx], y_scaled[idx], scaling
                        )
                        bootstrap_nmse.append(nmse_val)
                        results_matrix[dataset_idx, target_idx, bootstrap_idx] = nmse_val

                    mean_nmse = np.nanmean(bootstrap_nmse)
                    std_nmse = np.nanstd(bootstrap_nmse)

                    results_metadata.append({
                        "dataset_idx": dataset_idx, "dataset_name": dataset_info["name"],
                        "target_idx": target_idx, "target_name": target,
                        "equation": equation, "complexity": complexity,
                        "mean_nmse": mean_nmse, "std_nmse": std_nmse,
                        "model_path": model_path
                    })

                    summary_stats.append({
                        "dataset_idx": dataset_idx, "dataset_name": dataset_info["name"],
                        "target_idx": target_idx, "target_name": target,
                        "mean_nmse": mean_nmse, "std_nmse": std_nmse,
                        "min_nmse": np.nanmin(bootstrap_nmse),
                        "max_nmse": np.nanmax(bootstrap_nmse),
                        "n_valid": np.sum(~np.isnan(bootstrap_nmse))
                    })

                    print(f"    NMSE: {mean_nmse:.4f} ± {std_nmse:.4f}")

                except Exception as e:
                    print(f"  Error evaluating {target}: {e}")
                    continue

        # === Save results for this run ===
        np.save(f"{output_dir}/{label}_nmse_matrix_3x2x{n_bootstrap}.npy", results_matrix)

        # Flattened CSV
        flattened_results = []
        for dataset_idx in range(3):
            for target_idx in range(2):
                for bootstrap_idx in range(n_bootstrap):
                    flattened_results.append({
                        "dataset_idx": dataset_idx,
                        "dataset_name": datasets[dataset_idx]["name"],
                        "target_idx": target_idx,
                        "target_name": targets[target_idx],
                        "bootstrap_idx": bootstrap_idx,
                        "bootstrap_seed": 42 + bootstrap_idx,
                        "nmse": results_matrix[dataset_idx, target_idx, bootstrap_idx]
                    })
        pd.DataFrame(flattened_results).to_csv(f"{output_dir}/{label}_nmse_matrix_flattened.csv", index=False)

        # Wide-format CSV
        wide_results = []
        for dataset_idx in range(3):
            for target_idx in range(2):
                row = {
                    "dataset_idx": dataset_idx,
                    "dataset_name": datasets[dataset_idx]["name"],
                    "target_idx": target_idx,
                    "target_name": targets[target_idx]
                }
                vals = results_matrix[dataset_idx, target_idx, :]
                for bootstrap_idx in range(n_bootstrap):
                    row[f"bootstrap_{bootstrap_idx}"] = vals[bootstrap_idx]
                row["mean_nmse"] = np.nanmean(vals)
                row["std_nmse"] = np.nanstd(vals)
                row["n_valid"] = np.sum(~np.isnan(vals))
                wide_results.append(row)
        pd.DataFrame(wide_results).to_csv(f"{output_dir}/{label}_nmse_matrix_wide_format.csv", index=False)

        # Metadata & summary
        pd.DataFrame(results_metadata).to_csv(f"{output_dir}/{label}_evaluation_metadata.csv", index=False)
        pd.DataFrame(summary_stats).to_csv(f"{output_dir}/{label}_evaluation_summary.csv", index=False)

        print(f"\nResults saved for {label.upper()} in {output_dir}")
        return results_matrix, pd.DataFrame(results_metadata), pd.DataFrame(summary_stats)

    # === Run separately for train and val ===
    train_results = evaluate_and_save(train_datasets, "train")
    val_results = evaluate_and_save(val_datasets, "val")

    return {"train": train_results, "val": val_results}


def evaluate_single_equation_scaled(equation_str, X_scaled, y_scaled, scaling):
    """
    Evaluate a single equation on scaled features and targets.

    Parameters
    ----------
    equation_str : str
        Equation from model_equations.csv, in sympy_format.
    X_scaled : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    y_scaled : np.ndarray
        Scaled target values (n_samples,).
    scaling : dict
        Scaling parameters (unused here but passed for consistency).

    Returns
    -------
    float
        Normalized Mean Squared Error (NMSE).
    """
    try:
        # Define symbols for 6 features
        symbols = sp.symbols("x0 x1 x2 x3 x4 x5")

        # Parse equation string
        expr = sp.sympify(equation_str)

        # Turn into a NumPy-aware function
        func = sp.lambdify(symbols, expr, "numpy")

        # Ensure X_scaled has correct shape (n_samples, 6)
        if X_scaled.shape[1] != len(symbols):
            raise ValueError(
                f"Feature mismatch: equation expects {len(symbols)} inputs, "
                f"but got {X_scaled.shape[1]} features."
            )

        # Evaluate predictions
        y_pred = func(*[X_scaled[:, i] for i in range(X_scaled.shape[1])])

        # Flatten in case func returns weird shape
        y_pred = np.asarray(y_pred).reshape(-1)

        # Compute NMSE
        mse = np.mean((y_scaled - y_pred) ** 2)
        var = np.var(y_scaled)
        nmse = mse / var if var > 0 else np.nan

        return nmse

    except Exception as e:
        print(f"    Error in evaluate_single_equation_scaled: {e}")
        return np.nan


def extract_terms_and_ops(expr_str):
    """Parse a sympy expression string and return operators, terms, and constants."""
    try:
        expr = sp.sympify(expr_str)
    except Exception:
        return [], [], []
    ops, terms, consts = set(), set(), set()
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Symbol):
            terms.add(str(node))

        
        #elif isinstance(node, sp.Number):
        #    consts.add(float(node))
        #elif isinstance(node, sp.Function):
        #    ops.add(str(node.func))
        #elif isinstance(node, sp.Add):
        #    ops.add('+')
        #elif isinstance(node, sp.Mul):
        #    ops.add('*')
        #elif isinstance(node, sp.Pow):
        #    ops.add('^')
        
        # Division is represented as Mul with negative powers in sympy,
        # but you can also check for division by looking for Pow with negative exponent.
        # If you want to explicitly count division, you can do:
        #if isinstance(node, sp.Pow) and node.args[1] == -1:
        #    ops.add('/')
    return ops, terms, consts

def analyze_equation_terms_by_complexity(base_directory, use_sympy_format=True):
    """
    Scans all model_equations.csv files and analyzes operator/term frequency by complexity and dataset.
    Plots bar charts and word clouds with normalized frequencies.
    """
    # Variable translation function (same as in analyze_best_equation_terms_by_dataset)
    def translate_var(var):
        mapping = {
            "x0": "gamma",
            "x1": "lambda",
            "x2": "delta",
            "x3": "epsilon",
            "x4": "N1",
            "x5": "N2"
        }
        return mapping.get(var, var)
    
    # Dataset title mapping function
    def get_readable_title(dataset_name):
        title_mapping = {
            "flux_only_bias_train_e3": "Bias, Flux",
            "flux_only_nobias_train_e3": "Wide Tau, Flux", 
            "flux_only_nobias_train_e3_narrowtau": "Narrow Tau, Flux",
            "w1_only_bias_train_e3": "Bias, W1",
            "w1_only_nobias_train_e3": "Wide Tau, W1",
            "w1_only_nobias_train_e3_narrowtau": "Narrow Tau, W1"
        }
        return title_mapping.get(dataset_name, dataset_name)
    
    complexity_term_counter = defaultdict(Counter)
    dataset_term_counter = defaultdict(Counter)
    complexity_equation_counts = defaultdict(int)  # Track total equations per complexity
    dataset_equation_counts = defaultdict(int)     # Track total equations per dataset
    all_terms_ops = Counter()
    file_info = []

    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file == "model_equations.csv":
                csv_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                dataset_name = os.path.basename(os.path.dirname(root))
                for _, row in df.iterrows():
                    complexity = row['complexity']
                    expr_str = row[4] if use_sympy_format else row[2]
                    ops, terms, consts = extract_terms_and_ops(expr_str)
                    
                    # Count equations for normalization
                    complexity_equation_counts[complexity] += 1
                    dataset_equation_counts[dataset_name] += 1
                    
                    # Aggregate
                    for item in list(ops) + list(terms):
                        complexity_term_counter[complexity][item] += 1
                        dataset_term_counter[dataset_name][item] += 1
                        all_terms_ops[item] += 1
                file_info.append((dataset_name, csv_path))

    # Define consistent operator order (alphabetical with variables first)
    def get_operator_order(items):
        variables = sorted([item for item in items if item.startswith('x')])
        operators = sorted([item for item in items if not item.startswith('x')])
        return variables + operators

    # Get all unique operators/terms for consistent ordering and create color mapping
    all_items = set()
    for counter in complexity_term_counter.values():
        all_items.update(counter.keys())
    for counter in dataset_term_counter.values():
        all_items.update(counter.keys())
    ordered_items = get_operator_order(all_items)
    
    # Create consistent color mapping for all operators
    colors = cm.Set3(np.linspace(0, 1, len(ordered_items)))
    color_map = {item: colors[i] for i, item in enumerate(ordered_items)}

    # DATASET CHARTS: Each dataset gets its own separate chart with colored bars for each operator
    # Define the specific pairing order you want
    dataset_pairs = [
        ("flux_only_bias_train_e3", "w1_only_bias_train_e3"),
        ("flux_only_nobias_train_e3", "w1_only_nobias_train_e3"), 
        ("flux_only_nobias_train_e3_narrowtau", "w1_only_nobias_train_e3_narrowtau")
    ]

    # Filter to only include datasets that actually exist in your data
    existing_datasets = set(dataset_term_counter.keys())
    valid_pairs = [(d1, d2) for d1, d2 in dataset_pairs 
                if d1 in existing_datasets or d2 in existing_datasets]

    if valid_pairs:
        n_rows = len(valid_pairs)
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for row, (dataset1, dataset2) in enumerate(valid_pairs):
            for col, dataset in enumerate([dataset1, dataset2]):
                if dataset not in existing_datasets:
                    # Hide this subplot if dataset doesn't exist
                    if n_rows > 1:
                        axes[row, col].set_visible(False)
                    else:
                        axes[col].set_visible(False)
                    continue
                    
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                counter = dataset_term_counter[dataset]
                total_equations = dataset_equation_counts[dataset]
                
                # Get only the operators that actually appear in this dataset and translate variables
                for item in ordered_items:
                    raw_count = counter.get(item, 0)
                    if raw_count > 0 and raw_count < 3:  # tiny counts
                        print(f"Suspicious count: {item} -> {raw_count} in {dataset}")
                appearing_items = ordered_items
                translated_items = [translate_var(item) for item in appearing_items]
                normalized_freqs = [
                    counter.get(item, 0) / total_equations if total_equations > 0 else 0 
                    for item in appearing_items
                    
                ]
                bar_colors = [color_map[item] for item in appearing_items]
                
                # Make bars half width
                bars = ax.bar(translated_items, normalized_freqs, color=bar_colors, width=0.4)
                ax.set_title(get_readable_title(dataset), fontsize=16)
                
                # Add value labels on top or inside bars
                for bar, freq in zip(bars, normalized_freqs):
                    xpos = bar.get_x() + bar.get_width()/2
                    if freq < 0.15:
                        ax.text(xpos, freq + 0.01, f"{freq:.2f}", ha='center', va='bottom', fontsize=10)
                    else:
                        ax.text(xpos, freq/2, f"{freq:.2f}", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
                
                # Make spacing tighter by adjusting x-axis limits
                if len(translated_items) > 0:
                    ax.set_xlim(-0.6, len(translated_items) - 0.4)
                
                # Only show y-axis label on middle left subplot
                if col == 0 and row == len(valid_pairs) // 2:
                    ax.set_ylabel("Normalized Frequency", fontsize=16)
                
                ax.tick_params(axis='x', rotation=0, labelsize=16)
                ax.tick_params(axis='y', labelsize=16)
                ax.grid(True, alpha=0.3)

        plt.suptitle("Normalized Variable Frequency by Dataset")
        plt.tight_layout()
        plt.show()
        # Print file info and normalization stats
        print("Scanned files:")
        for dataset, path in file_info:
            print(f"{dataset}: {path}")
        
        print("\nNormalization info:")
        print("Equations per complexity:")
        for complexity in sorted(complexity_equation_counts.keys()):
            print(f"  Complexity {complexity}: {complexity_equation_counts[complexity]} equations")
        print("Equations per dataset:")
        for dataset in sorted(dataset_equation_counts.keys()):
            print(f"  {dataset}: {dataset_equation_counts[dataset]} equations")
        
if __name__ == '__main__':
    # --- Configuration ---
    # Define the base directory where all your run folders are located.
    # You should have 6 folders in total for the PySR runs.
    # Example: 'C:/Users/brand/Desktop/My_PySR_Runs/'
    all_runs_base_dir = r'C:/Users/brand/Desktop/Raj-Sindi/training_data/sim_csv_v11/'
    train_dir = os.path.join(all_runs_base_dir, 'train')
    val_dir = os.path.join(all_runs_base_dir, 'val')

    csv_folder = r"C:\Users\brand\Desktop\School\Quals\data\flux_only_bias_12hr 5-set"
    #plot_nmse_vs_complexity_from_subdirs(csv_folder)

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
    output_visualization_dir = os.path.join(r"C:\Users\brand\Desktop\School\Quals", "Presentation_Visualizations")
    os.makedirs(output_visualization_dir, exist_ok=True)
    
    # --- Generate Visualizations ---
    
    # 1. Generate "The PySR Gauntlet" plot
    print("--- Generating PySR Performance Comparison Plot ---")
    #plot_pysr_performance_comparison(pysr_run_directories, output_visualization_dir)
    
    # 2. Generate "The Method Showdown" plot
    #print("\n--- Generating Method Showdown Plot ---")
    # For this comparison, we'll use the best PySR result from the most realistic dataset:
    # Biased, Wide-Range Tau, for the FValue target.
    #best_pysr_run_path = pysr_run_directories['Flux']['Biased Wide Tau']
    #best_pysr_nmse_value = get_best_pysr_nmse(best_pysr_run_path)
    
    #if best_pysr_nmse_value is not None:
        #plot_method_showdown(best_pysr_nmse_value, output_visualization_dir)
    #    print("Temp")
    #else:
    #   print("Could not generate Method Showdown plot because the best PySR NMSE could not be determined.")

    # NMSE Scouring Directory
    nmse_scour_directory = r'C:\Users\brand\Desktop\School\Quals\data\train'
    flux_bias_directory = nmse_scour_directory + r'\flux_only_bias_train_e3'
    w1_bias_directory = nmse_scour_directory + r'\w1_only_bias_train_e3'
    flux_nobias_directory = nmse_scour_directory + r'\flux_only_nobias_train_e3'
    w1_nobias_directory = nmse_scour_directory + r'\w1_only_nobias_train_e3'
    flux_nobias_narrow_directory = nmse_scour_directory + r'\flux_only_nobias_train_e3_narrowtau'
    w1_nobias_narrow_directory = nmse_scour_directory + r'\w1_only_nobias_train_e3_narrowtau'
    all_directories = [
        flux_bias_directory,
        flux_nobias_directory,
        flux_nobias_narrow_directory,
        w1_bias_directory,
        w1_nobias_directory,
        w1_nobias_narrow_directory
    ]

    #analyze_nmse_by_complexity_subplots(all_directories)
    #analyze_nmse_by_complexity_with_inset(nmse_scour_directory)

    # results_matrix, metadata_df, summary_df = evaluate_models_on_datasets_structured(
    #     output_dir=f"{output_visualization_dir}/nmse_box_train", 
    #     n_bootstrap=1000
    # )


    analyze_equation_terms_by_complexity(train_dir, use_sympy_format=True)

    


    # Analyze best equations only (variable/operator frequency and table)
    # analyze_best_equation_terms_by_dataset(
    #     base_directory="C:/Users/brand/Desktop/School/Quals/data/train",
    #     save_dir=output_visualization_dir
    # )


