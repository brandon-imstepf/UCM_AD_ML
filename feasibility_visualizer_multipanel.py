import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import re
from pysr_analysis_utils import setup_matplotlib_settings, find_files, extract_terms_and_ops, translate_var, get_readable_title
from feasibility_config import GROUND_TRUTH_FUNCS, BEST_RUN_PATHS, get_multipanel_noise_levels, OUTPUT_PLOT_DIR

# Setup consistent plotting styles
setup_matplotlib_settings()

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

def get_data_and_equation(model_name, noise_level):
    """
    Loads data and PySR equation for a given model and noise level.
    Returns (x_data, y_data, pysr_expr, pysr_func, sampling_method, domain_info) or (None, None, None, None, None, None) on failure.
    Handles both 1D and multi-dimensional functions.
    """
    pysr_run_dir = BEST_RUN_PATHS.get(model_name, {}).get(str(noise_level))
    if not pysr_run_dir:
        return None, None, None, None, None, None
    
    if not os.path.isdir(pysr_run_dir):
        return None, None, None, None, None, None

    # Infer dataset path from the run path
    path_parts = pysr_run_dir.split(os.sep)
    dataset_name_part = next((part for part in reversed(path_parts) if model_name in part), None)
    
    if not dataset_name_part:
        return None, None, None, None, None, None

    # Find the dataset CSV robustly
    noise_dir = os.path.dirname(os.path.dirname(pysr_run_dir))
    dataset_path = os.path.join(noise_dir, f"{dataset_name_part}.csv")
    if not os.path.exists(dataset_path):
        noise_dir = os.path.dirname(noise_dir)
        dataset_path = os.path.join(noise_dir, f"{dataset_name_part}.csv")
        if not os.path.exists(dataset_path):
            return None, None, None, None, None, None
        
    try:
        data_df = pd.read_csv(dataset_path)
        # Get number of dimensions from the model
        _, _, num_dims = GROUND_TRUTH_FUNCS.get(model_name, (None, None, 1))
        # For multi-dimensional functions, use all columns; for 1D, just use first
        if num_dims > 1:
            x_data = data_df.iloc[:, :num_dims]
        else:
            x_data = data_df.iloc[:, 0]
        y_data = data_df.iloc[:, -1]
    except Exception as e:
        print(f"[WARNING] Could not read dataset {dataset_path}: {e}")
        return None, None, None, None, None, None

    pysr_eq_str = get_best_pysr_equation(pysr_run_dir)
    
    pysr_expr = None
    pysr_func = None
    if pysr_eq_str:
        try:
            # Get number of dimensions for this model
            _, _, num_dims = GROUND_TRUTH_FUNCS.get(model_name, (None, None, 1))
            
            # Create symbols for all dimensions
            if num_dims == 1:
                symbols_list = sp.symbols('x0')
            else:
                symbols_list = sp.symbols(f'x0:{num_dims}')
            
            pysr_expr = sp.sympify(pysr_eq_str)
            # lambdify handles both single symbol and list of symbols
            pysr_func = sp.lambdify(symbols_list, pysr_expr, 'numpy')
        except Exception as e:
            print(f"[WARNING] Error parsing PySR equation: {e}")
            return x_data, y_data, None, None, None, None

    # Extract sampling method from path (e.g., "LHS" or "RandSample")
    sampling_method = "Unknown"
    for part in pysr_run_dir.split(os.sep):
        if "LHS" in part:
            sampling_method = "LHS"
            break
        elif "RandSample" in part:
            sampling_method = "Random"
            break
    
    # Extract domain info from data
    if isinstance(x_data, pd.DataFrame):
        x_min = float(x_data.iloc[:, 0].min())
        x_max = float(x_data.iloc[:, 0].max())
    else:
        x_min = float(x_data.min())
        x_max = float(x_data.max())
    n_samples = len(x_data)

    return x_data, y_data, pysr_expr, pysr_func, sampling_method, (x_min, x_max, n_samples)


def plot_multipanel_feasibility(model_name, output_dir):
    """
    Creates a multipanel figure for a model across available noise levels.
    Automatically adjusts grid layout based on number of available noise levels.
    - Up to 3 levels: 1x3 grid
    - Up to 6 levels: 2x3 grid
    - More than 6 levels: 3x3 grid (or adjusts as needed)
    Handles both 1D and multi-dimensional functions.
    """
    ground_truth_func, ground_truth_str, num_dims = GROUND_TRUTH_FUNCS.get(model_name, (None, None, 1))
    if not ground_truth_func:
        print(f"[ERROR] Unknown model: {model_name}")
        return

    # Get noise levels dynamically for this model
    noise_levels = get_multipanel_noise_levels(model_name)
    if not noise_levels:
        print(f"[WARNING] No noise levels found for {model_name}, skipping multipanel.")
        return
    
    # Determine grid layout based on number of noise levels
    n_levels = len(noise_levels)
    if n_levels <= 3:
        rows, cols = 1, 3
        figsize = (18, 5)
    elif n_levels <= 6:
        rows, cols = 2, 3
        figsize = (18, 10)
    else:
        # For more than 6 levels, use 3x3 or larger
        cols = 3
        rows = (n_levels + cols - 1) // cols
        figsize = (18, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = axes.flatten() if rows > 1 or cols > 1 else [axes]
    
    # Get domain and sampling info from first noise level (same for all)
    first_result = get_data_and_equation(model_name, noise_levels[0])
    sampling_method = first_result[4] if first_result[4] else "Unknown"
    domain_info = first_result[5] if first_result[5] else (0, 20, 20)
    x_min, x_max, n_samples = domain_info
    
    # For multi-dimensional data, pre-compute all residuals to get shared y-axis range
    shared_y_min, shared_y_max = None, None
    if num_dims > 1:
        all_residuals = []
        for noise_level in noise_levels:
            result = get_data_and_equation(model_name, noise_level)
            if result[0] is not None:
                x_data, y_data, _, _, _, _ = result
                # Compute true y values
                y_true = []
                for idx in range(len(x_data)):
                    inputs = [x_data.iloc[idx, i] for i in range(num_dims)]
                    y_true.append(ground_truth_func(*inputs))
                y_true = np.array(y_true)
                residuals = y_data - y_true
                all_residuals.extend(residuals)
        
        if all_residuals:
            all_residuals = np.array(all_residuals)
            # Add 10% margin to the range
            margin = (all_residuals.max() - all_residuals.min()) * 0.1
            shared_y_min = all_residuals.min() - margin
            shared_y_max = all_residuals.max() + margin
    
    for idx, noise_level in enumerate(noise_levels):
        ax = axes_flat[idx]
        
        result = get_data_and_equation(model_name, noise_level)
        if result[0] is None:
            ax.text(0.5, 0.5, f'No data for\n{noise_level}% noise', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        x_data, y_data, pysr_expr, pysr_func, sampling_method, domain_info = result
        
        # Plot ground truth (for 1D only; for multi-D we'll use residuals)
        x_smooth = np.linspace(x_data.min() if isinstance(x_data, pd.Series) else x_data.iloc[:, 0].min(), 
                               x_data.max() if isinstance(x_data, pd.Series) else x_data.iloc[:, 0].max(), 400)
        
        # For 1D functions, plot the ground truth curve
        if num_dims == 1:
            y_ground_truth = ground_truth_func(x_smooth)
            ax.plot(x_smooth, y_ground_truth, 'k--', linewidth=2.5, zorder=2, label='True')
        # For multi-D, we'll show residuals instead (no curve needed here)
        
        # Plot recovered function if available and capture LaTeX for display
        recovered_latex = None
        if pysr_func:
            # For multi-dimensional functions, replicate x_smooth to match dimensions
            if num_dims > 1:
                y_pysr_raw_vals = []
                for xs in x_smooth:
                    # Create array of inputs: x_smooth value for x0, mean values for other dims
                    inputs = [xs] + [float(x_data.iloc[:, i].mean()) for i in range(1, num_dims)]
                    y_pysr_raw_vals.append(pysr_func(*inputs))
                y_pysr_raw = np.array(y_pysr_raw_vals)
            else:
                y_pysr_raw = pysr_func(x_smooth)
            
            if np.isscalar(y_pysr_raw) or (hasattr(y_pysr_raw, 'shape') and y_pysr_raw.shape == ()): 
                y_pysr = np.full_like(x_smooth, float(y_pysr_raw))
            else:
                y_pysr = y_pysr_raw
            
            # Only plot recovered line for 1D data; for multi-D, we'll show it but not as a curve
            if num_dims == 1:
                ax.plot(x_smooth, y_pysr, 'b-', linewidth=2, zorder=1, label='Recovered')
            # For multi-D, recovered function will be shown only in legend/title
            
            try:
                # pysr_expr is a sympy expression when available
                if pysr_expr is not None:
                    recovered_latex = sp.latex(pysr_expr)
            except Exception:
                recovered_latex = str(pysr_expr)

        # Plot data and residuals based on dimensionality
        if num_dims > 1:
            # For multi-dimensional data: plot residuals (difference from true function)
            # Compute true y values at actual data points
            y_true = []
            for idx in range(len(x_data)):
                inputs = [x_data.iloc[idx, i] for i in range(num_dims)]
                y_true.append(ground_truth_func(*inputs))
            y_true = np.array(y_true)
            
            # Residuals are the difference between actual y and true y
            residuals = y_data - y_true
            
            # Plot residuals vs first dimension (x0)
            x_plot = x_data.iloc[:, 0]
            ax.axhline(y=0, color='k', linestyle='--', linewidth=2.5, zorder=2, label='True (zero residual)')
            ax.scatter(x_plot, residuals, alpha=0.7, color='red', marker='.', s=50, zorder=3, label='Residuals')
            ax.set_ylabel('Residual (Data - True)', fontsize=12)
            
            # Apply shared y-axis range across all subplots
            if shared_y_min is not None and shared_y_max is not None:
                ax.set_ylim(shared_y_min, shared_y_max)
            
        else:
            # For 1D data: plot actual function curves and data points
            x_plot = x_data.iloc[:, 0] if isinstance(x_data, pd.DataFrame) else x_data
            ax.scatter(x_plot, y_data, alpha=0.7, color='red', marker='.', s=50, zorder=3, label='Noisy Data')
            ax.set_ylabel('y', fontsize=12)

        # Add noise percentage text in corner (larger font)
        ax.text(0.05, 0.95, f'{noise_level}% Noise', transform=ax.transAxes, 
               fontsize=28, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add legend only to the first subplot
        if idx == 0:
            ax.legend(loc='lower right', fontsize=14, framealpha=0.95)
        
        # Use the recovered equation as the subplot title (LaTeX when possible)
        if recovered_latex:
            display_latex = recovered_latex
            # Truncate very long LaTeX strings for aesthetics
            if len(display_latex) > 240:
                display_latex = display_latex[:237] + '...'
            try:
                ax.set_title(f'${display_latex}$', fontsize=18, pad=8)
            except Exception:
                ax.set_title(display_latex, fontsize=18, pad=8)
        else:
            # leave an empty title to keep consistent spacing
            ax.set_title('', fontsize=18, pad=8)
        
        # Remove tick labels for clean appearance
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a subtle grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.3)
        
        print(f"[OK] Plotted {model_name} at {noise_level}% noise")
    
    # Hide any unused subplots
    for idx in range(len(noise_levels), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    # (Removed main title above the subfigures)
    # Add ground-truth function on top line, domain info on second line below panels
    try:
        gt_latex = sp.latex(sp.sympify(ground_truth_str))
        true_str = f'True: ${gt_latex}$'
    except Exception:
        true_str = f'True: {ground_truth_str}'
    
    # Add domain and sampling info
    domain_str = f"Domain: $0 \\leq x \\leq {x_max:.0f}$, N = {n_samples}, {sampling_method} Sampling"
    
    # Place true function on top line, domain info on second line
    fig.text(0.3, 0.95, true_str, ha='center', va='center', fontsize=26)
    fig.text(0.7, 0.95, domain_str, ha='center', va='center', fontsize=26)

    # Leave more top margin to prevent clipping of LaTeX/title
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    output_filename = f"{model_name}_multipanel_feasibility.png"
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved multipanel plot to {save_path}")
    plt.close()


if __name__ == '__main__':
    # --- Configuration ---
    output_plot_dir = OUTPUT_PLOT_DIR
    os.makedirs(output_plot_dir, exist_ok=True)

    # --- Main Execution ---
    models_to_plot = list(GROUND_TRUTH_FUNCS.keys())
    for model_name in models_to_plot:
        print(f"\nGenerating multipanel figure for {model_name}...")
        plot_multipanel_feasibility(model_name, output_plot_dir)
    
    print("\n--- Multipanel feasibility visualization complete. ---")
