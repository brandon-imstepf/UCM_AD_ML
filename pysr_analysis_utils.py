import os
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd  # <--- Added this, required for data loading

# --- Matplotlib Settings ---
def setup_matplotlib_settings():
    """Sets up consistent matplotlib settings for plots."""
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

# --- File System Utilities ---
def find_files(base_dir, file_name):
    """
    Recursively finds all files with a given name in a directory.
    """
    found_files = []
    for root, _, files in os.walk(base_dir):
        if file_name in files:
            found_files.append(os.path.join(root, file_name))
    return found_files

def find_and_load_model_data(base_dir, known_datasets=None):
    """
    Scours subdirectories of a base directory to find and load PySR model data.
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

                if len(eq_df) != len(metrics_df):
                    print(f"⚠️ Warning: Row count mismatch in {root}. Skipping directory.")
                    continue

                dataset_name = None
                if known_datasets:
                    path_parts = root.split(os.sep)
                    for part in path_parts:
                        if part in known_datasets:
                            dataset_name = part
                            break

                combined_df = pd.DataFrame({
                    'complexity': pd.to_numeric(eq_df.get('complexity'), errors='coerce'),
                    'equation': eq_df.get('equation'),
                    'sympy_format': eq_df.get('sympy_format'),
                    'NMSE_Train': pd.to_numeric(metrics_df.get('NMSE_Train'), errors='coerce'),
                    'NMSE_Test': pd.to_numeric(metrics_df.get('NMSE_Test'), errors='coerce'),
                    'NMSE_Validation': pd.to_numeric(metrics_df.get('NMSE_Validation'), errors='coerce'),
                    'run_id': os.path.basename(root),
                    'dataset': dataset_name
                })
                
                all_model_data.append(combined_df)

            except Exception as e:
                print(f"❌ Error processing directory {root}: {e}")
                continue

    if not all_model_data:
        print("No valid model data was found.")
        return pd.DataFrame()

    final_df = pd.concat(all_model_data, ignore_index=True)
    final_df.dropna(subset=['complexity'], inplace=True)
    final_df = final_df.dropna(subset=['NMSE_Train', 'NMSE_Test', 'NMSE_Validation'], how='all')
    
    print(f"✅ Successfully loaded and combined data from {len(all_model_data)} runs.")
    return final_df

# --- Sympy Utilities ---
def extract_terms_and_ops(expr_str):
    """Parse a sympy expression string and return operators, terms, and constants."""
    try:
        expr = sp.sympify(expr_str)
    except (sp.SympifyError, TypeError, AttributeError):
        return set(), set(), set()

    ops = set()
    terms = {str(s) for s in expr.free_symbols}

    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Function):
            ops.add(node.func.__name__)
        elif node.is_Add:
            ops.add('add')
        elif node.is_Mul:
            ops.add('mul')
        elif node.is_Pow:
            ops.add('pow')

    consts = set()
    return ops, terms, consts

# --- Plotting & Naming Utilities ---
def translate_var(var):
    """Translates variable names from x# to descriptive names."""
    mapping = {
        "x0": "γ", "x1": "λ", "x2": "δ",
        "x3": "ε", "x4": "N1", "x5": "N2"
    }
    return mapping.get(var, var)

def get_readable_title(dataset_name):
    """Generates a readable title from a dataset directory name."""
    title_mapping = {
        "flux_only_bias_train_e3": r"Bias, $\Phi_{\tau}$",
        "flux_only_nobias_train_e3": r"No Bias, $\Phi_{\tau}$",
        "flux_only_nobias_train_e3_narrowtau": r"Narrow Tau, $\Phi_{\tau}$",
        "w1_only_bias_train_e3": r"Bias, $\Delta_{\tau}$",
        "w1_only_nobias_train_e3": r"No Bias, $\Delta_{\tau}$",
        "w1_only_nobias_train_e3_narrowtau": r"Narrow Tau, $\Delta_{\tau}$"
    }
    return title_mapping.get(dataset_name, dataset_name)