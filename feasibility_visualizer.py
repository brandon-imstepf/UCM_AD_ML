import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import re
from regression_benchmark_functions import *
from pysr_analysis_utils import setup_matplotlib_settings, find_files, extract_terms_and_ops, translate_var, get_readable_title

# Setup consistent plotting styles
setup_matplotlib_settings()

# --- Ground Truth Function Mapping ---
GROUND_TRUTH_FUNCS = {
    "Nguyen_F1": (Nguyen_F1, "x**3 + x**2 + x"),
    "Nguyen_F2": (Nguyen_F2, "x**4 + x**3 + x**2 + x"),
}

# --- Manually specified "best" run directories ---
BEST_RUN_PATHS = {
    "Nguyen_F1": {
        "0": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\0_Percent_Noise\0_Percent_Noise\Nguyen_F1_LHS\2025-11-25_11-12_5400_Info",
        "1": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\1_Percent_Noise\1_Percent_Noise\Nguyen_F1_LHS\2025-12-01_13-16_5400_Info",
        "3": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\3_Percent_Noise\3_Percent_Noise\Nguyen_F1_RandSample\2025-11-25_11-14_5400_Info",
        "5": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\5_Percent_Noise\5_Percent_Noise\Nguyen_F1_LHS\2025-12-01_13-40_5400_Info",
        "10": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\10_Percent_Noise\10_Percent_Noise\Nguyen_F1_LHS\2025-11-25_11-12_5400_Info",
        "25": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\25_Percent_Noise\25_Percent_Noise\Nguyen_F1_LHS\2025-11-25_11-13_5400_Info",
    },
    "Nguyen_F2": {
        "0": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\0_Percent_Noise\0_Percent_Noise\Nguyen_F2_LHS\2025-11-25_11-16_5400_Info",
        "1": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\1_Percent_Noise\1_Percent_Noise\Nguyen_F2_LHS\2025-11-25_11-16_5400_Info",
        "3": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\3_Percent_Noise\3_Percent_Noise\Nguyen_F2_LHS\2025-11-25_11-16_5400_Info",
        "5": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\5_Percent_Noise\5_Percent_Noise\Nguyen_F2_LHS\2025-11-25_11-16_5400_Info",
        "10": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\10_Percent_Noise\10_Percent_Noise\Nguyen_F2_LHS\2025-11-25_11-16_5400_Info",
        "25": r"C:\Users\brand\Desktop\School\Quals\data\Nguyen-RBFs-n-ns-(1,3,40)_fixedNoise\25_Percent_Noise\25_Percent_Noise\Nguyen_F2_LHS\2025-11-25_11-16_5400_Info",
    }
}

def get_best_pysr_equation(pysr_run_dir):
    """Finds and extracts the best equation from a PySR run's best.txt file or hall_of_fame.csv."""
    if not pysr_run_dir or not os.path.isdir(pysr_run_dir):
        return None

    best_txt_path = os.path.join(pysr_run_dir, 'best.txt')
    if os.path.exists(best_txt_path):
        with open(best_txt_path, 'r') as f:
            for line in f:
                if "Selected Best Function:" in line:
                    match = re.search(r"Selected Best Function:\s*\d+,\s*(.*)", line)
                    if match:
                        return match.group(1).strip()

    hof_csv_path = os.path.join(pysr_run_dir, 'hall_of_fame.csv')
    if os.path.exists(hof_csv_path):
        try:
            hof_df = pd.read_csv(hof_csv_path)
            if not hof_df.empty:
                return hof_df.iloc[-1]['Equation']
        except Exception as e:
            print(f"[WARNING] Warning: Could not read hall_of_fame file {hof_csv_path}: {e}")
    
    return None

def plot_1d_feasibility_comparison(model_name, noise_level, output_dir):
    """
    Generates and saves a 1D plot with specified colors and styles using manual paths.
    """
    ground_truth_func, ground_truth_str = GROUND_TRUTH_FUNCS.get(model_name, (None, None))
    if not ground_truth_func:
        return

    pysr_run_dir = BEST_RUN_PATHS.get(model_name, {}).get(str(noise_level))
    if not pysr_run_dir:
        print(f"[INFO] Info: No manual path specified for {model_name} at {noise_level}% noise. Skipping.")
        return
    
    if not os.path.isdir(pysr_run_dir):
        print(f"[ERROR] Error: PySR run directory does not exist: {pysr_run_dir}")
        return

    # Infer dataset path from the run path
    path_parts = pysr_run_dir.split(os.sep)
    dataset_name_part = next((part for part in reversed(path_parts) if model_name in part), None)
    
    if not dataset_name_part:
        print(f"[WARNING] Warning: Could not determine dataset name from path: {pysr_run_dir}")
        return

    # Find the dataset CSV robustly
    noise_dir = os.path.dirname(os.path.dirname(pysr_run_dir)) # Go up two levels from the Info dir
    dataset_path = os.path.join(noise_dir, f"{dataset_name_part}.csv")
    if not os.path.exists(dataset_path):
         # Try one more level up for inconsistent structures
        noise_dir = os.path.dirname(noise_dir)
        dataset_path = os.path.join(noise_dir, f"{dataset_name_part}.csv")
        if not os.path.exists(dataset_path):
            print(f"[WARNING] Warning: Dataset for {dataset_name_part} not found near {pysr_run_dir}. Skipping.")
            return
        
    data_df = pd.read_csv(dataset_path)
    x_data = data_df.iloc[:, 0]
    y_data = data_df.iloc[:, -1]

    pysr_eq_str = get_best_pysr_equation(pysr_run_dir)
    
    if not pysr_eq_str:
        print(f"[WARNING] Warning: No PySR equation found for {model_name} at {noise_level}% noise in {pysr_run_dir}")
        print(f"  Files in directory: {os.listdir(pysr_run_dir) if os.path.isdir(pysr_run_dir) else 'N/A'}")
    
    pysr_func = None
    if pysr_eq_str:
        try:
            x0 = sp.symbols('x0')
            pysr_expr = sp.sympify(pysr_eq_str)
            pysr_func = sp.lambdify(x0, pysr_expr, 'numpy')
        except Exception as e:
            print(f"[ERROR] Error parsing PySR equation '{pysr_eq_str}': {e}")

    plt.figure(figsize=(12, 8))
    
    x_smooth = np.linspace(x_data.min(), x_data.max(), 400)
    y_ground_truth = ground_truth_func(x_smooth)
    plt.plot(x_smooth, y_ground_truth, 'k--', linewidth=2.5, label=f'True Function: ${sp.latex(sp.sympify(ground_truth_str))}$', zorder=2)

    if pysr_func:
        y_pysr_raw = pysr_func(x_smooth)
        if np.isscalar(y_pysr_raw) or (hasattr(y_pysr_raw, 'shape') and y_pysr_raw.shape == ()):
            y_pysr = np.full_like(x_smooth, float(y_pysr_raw))
        else:
            y_pysr = y_pysr_raw
        plt.plot(x_smooth, y_pysr, 'b-', linewidth=2, label=f'Recovered Function: ${sp.latex(pysr_expr)}$', zorder=1)

    plt.scatter(x_data, y_data, alpha=0.7, label='Training Points', color='red', marker='.', zorder=3)

    plt.title(f'{model_name} with {noise_level}% Noise')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    output_filename = f"{model_name}_{noise_level}pct_noise_manual.png"
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Saved 1D plot to {save_path}")
    plt.close()


if __name__ == '__main__':
    # --- Configuration ---
    output_plot_dir = r'C:\Users\brand\Desktop\School\Quals\Presentation_Visualizations\feasibility_plots'
    os.makedirs(output_plot_dir, exist_ok=True)

    # --- Main Execution ---
    models_to_plot = ['Nguyen_F1', 'Nguyen_F2']
    for model_name in models_to_plot:
        # Get noise levels from the feasibility CSV to ensure all are covered
        feasibility_df = pd.read_csv(r'C:\Users\brand\Desktop\School\Quals\data\feasibility.csv')
        model_row = feasibility_df[feasibility_df['model'] == model_name]
        if not model_row.empty:
            for noise_col in feasibility_df.columns[1:]:
                score = model_row.iloc[0][noise_col]
                if pd.to_numeric(score, errors='coerce') >= 0:
                     # Extract numeric noise level from column name (e.g., "0" from "0%" or similar)
                     try:
                         noise_level = int(noise_col.replace('%', '').replace('_', '').strip())
                     except ValueError:
                         noise_level = noise_col
                     print(f"\nPlotting {model_name} at {noise_level}% noise...")
                     plot_1d_feasibility_comparison(
                         model_name=model_name,
                         noise_level=noise_level,
                         output_dir=output_plot_dir
                     )
    
    print("\n--- Feasibility visualization complete. ---")