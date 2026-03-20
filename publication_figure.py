import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, NullFormatter, FixedLocator, FixedFormatter, NullLocator

# --- Import your existing data loaders ---
from pysr_analysis_utils import find_and_load_model_data, setup_matplotlib_settings

def generate_publication_figure_v2(
    phi_dir, 
    delta_dir, 
    save_path,
    best_complexity_phi=23, 
    best_complexity_delta=28,
    save_filename="Final_PySR_Graph_v2.png"
):
    # 1. --- SETUP STYLE (Matched to your Regression-Graph.ipynb) ---
    plt.rcParams['font.family'] = "Times New Roman"
    plt.rcParams['font.serif'] = ["Times New Roman"]
    plt.rcParams['font.sans-serif'] = ["Times New Roman"]
    
    # --- FORCE MATH/LATEX TO USE TIMES NEW ROMAN STYLE ---
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # Colors from the ipynb (c1 and c2)
    c1 = [0, 0.4470, 0.7410]      # Blue (Training)
    c2 = [0.4660, 0.6740, 0.1880] # Green (Out of Sample Testing)
    
    bar_palette = ['#F9A8A8', '#A8D1F9', '#A8F9B6', '#E4A8F9', '#F9E4A8', '#D3D3D3']

    # 2. --- LOAD DATA ---
    df_phi = find_and_load_model_data(phi_dir)
    df_delta = find_and_load_model_data(delta_dir)

    if df_phi.empty or df_delta.empty:
        print("❌ Error: DataFrames are empty. Check your data paths.")
        return

    # 3. --- DEFINE LAYOUT ---
    fig = plt.figure(figsize=(20, 22)) 
    
    gs_a = gridspec.GridSpec(1, 2, figure=fig, left=0.12, right=0.98, top=0.96, bottom=0.69, wspace=0.15)
    gs_b = gridspec.GridSpec(1, 2, figure=fig, left=0.12, right=0.98, top=0.63, bottom=0.35, wspace=0.15)
    gs_c = gridspec.GridSpec(1, 2, figure=fig, left=0.12, right=0.98, top=0.32, bottom=0.20, wspace=0.15)

    ax_nmse_phi = fig.add_subplot(gs_a[0, 0])
    ax_nmse_delta = fig.add_subplot(gs_a[0, 1])
    
    ax_bar_phi = fig.add_subplot(gs_b[0, 0])
    ax_bar_delta = fig.add_subplot(gs_b[0, 1])
    
    ax_eq_phi = fig.add_subplot(gs_c[0, 0])
    ax_eq_delta = fig.add_subplot(gs_c[0, 1])

    # ---------------------------------------------------------
    # ROW A: NMSE (Fixed Labels, Data, and Scale)
    # ---------------------------------------------------------

    def plot_nmse(ax, df, best_comp, title_math, is_left=True):

        numeric_cols = ['complexity', 'NMSE_Train', 'NMSE_Test', 'NMSE_Validation']
        # Group and sort to ensure the graph is connected with no gaps
        grouped = df[numeric_cols].groupby('complexity').mean().reset_index()
        grouped = grouped.sort_values('complexity')

        # Remove any potential NaNs to prevent disjoint lines
        train_plot = grouped.dropna(subset=['NMSE_Train'])
        val_plot = grouped.dropna(subset=['NMSE_Validation'])

        # Plotting Training (Actual Train Data) and Out of Sample Testing (Validation Data)
        ax.plot(train_plot['complexity'], train_plot['NMSE_Train'], label='Training', color=c1, lw=3)
        ax.plot(val_plot['complexity'], val_plot['NMSE_Validation'], label='Validation', color=c2, lw=3)

        # Keep the "best" vertical line
        ax.axvline(x=best_comp, color='#D55E00', linestyle='--', lw=5, label='Average\n "Best"')

        # Formatting: Log scale and unified X-axis (0 to 60)
        ax.set_yscale('log')
        ax.set_xlim(0, 60)
        ax.set_xticks(range(0, 61, 10))
        
        # Manually sync Y-limits to ensure ticks align across both plots
        ax.set_ylim(0.09, 2) 

        # --- Handle Y-axis ticks and labels ---
        # 1. Define the exact numeric values for the ticks
        ticks = [1.0, 0.5, 0.2, 0.1]
        
        # 2. Define exactly how you want those values to read using LaTeX formatting
        labels = [r'$10^0$', r'$5\times10^{-1}$', r'$2\times10^{-1}$', r'$10^{-1}$']
        
        # Apply the physical tick locations to both plots
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())

        if is_left:
            # Apply the manual labels to the left plot
            ax.yaxis.set_major_formatter(FixedFormatter(labels))
            ax.tick_params(axis='y', which='major', labelsize=28, labelleft=True)
        else:
            # For the right plot: Keep the tick marks but hide the labels (text)
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.tick_params(axis='y', which='major', labelleft=False)

        ax.set_title(title_math, fontsize=36)
        ax.tick_params(axis='x', labelsize=36)

    # Call the functions with the new 'is_left' parameter
    plot_nmse(ax_nmse_phi, df_phi, best_complexity_phi, r'$\phi_{\tau}$', is_left=True)
    ax_nmse_phi.set_ylabel('Avg. NMSE (Log Scale)', fontsize=36)

    plot_nmse(ax_nmse_delta, df_delta, best_complexity_delta, r'$\Delta_{\tau}$', is_left=False)
    ax_nmse_delta.legend(fontsize=30, loc='upper right')

    # Common X Label for Row A - Using 0.55 to center relative to plots (accounting for margin)
    fig.text(0.55, 0.65, 'Complexity', ha='center', fontsize=36)
    
    # ---------------------------------------------------------
    # ROW B: Bars (Gridlines Removed)
    # ---------------------------------------------------------
    data_phi = [0.79, 0.00, 0.24, 0.04, 0.75, 0.82] 
    data_delta = [0.84, 0.57, 0.33, 0.17, 0.15, 0.00]
    vars_labels = [r'$\gamma$', r'$\lambda$', r'$\delta$', r'$\epsilon$', r'$N_1$', r'$N_2$']
    x_pos = np.arange(len(vars_labels))

    def plot_bars(ax, data, title_math):
        ax.bar(x_pos, data, color=bar_palette, alpha=0.9, edgecolor='black', linewidth=1)
        ax.set_ylim(0, 1.05)
        ax.grid(False) # Removed gridlines
        ax.set_title(title_math, fontsize=40)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(vars_labels, fontsize=36)
        ax.tick_params(axis='y', labelsize=32)

    plot_bars(ax_bar_phi, data_phi, r'$\phi_{\tau}$')
    ax_bar_phi.set_ylabel('Normalized Frequency', fontsize=35)
    plot_bars(ax_bar_delta, data_delta, r'$\Delta_{\tau}$')
    ax_bar_delta.set_yticklabels([])

    # ---------------------------------------------------------
    # ROW C: Equations (Matching Font Sizes)
    # ---------------------------------------------------------
    for ax in [ax_eq_phi, ax_eq_delta]: ax.axis('off')

    eq_phi_latex = r"$\phi_{\tau} = \frac{N_2 + N_1}{e^\gamma} \cdot 0.31 - \frac{0.48 + \gamma}{\gamma - N_2 + 0.77}$"
    ax_eq_phi.text(0.45, 0.5, eq_phi_latex, ha='center', va='center', fontsize=36, transform=ax_eq_phi.transAxes)

    eq_delta_latex = r"$\Delta_{\tau} = (1.58 \cdot e^\gamma)^{-(5.2+\delta+\lambda)} - 0.84^{(\lambda+\gamma-\epsilon)}$"
    ax_eq_delta.text(0.5, 0.5, eq_delta_latex, ha='center', va='center', fontsize=32, transform=ax_eq_delta.transAxes)

    # Label A, B, C
    fig.text(0.02, 0.96, "(A)", fontsize=48, fontweight='bold', va='top')
    fig.text(0.02, 0.63, "(B)", fontsize=48, fontweight='bold', va='top')
    fig.text(0.02, 0.32, "(C)", fontsize=48, fontweight='bold', va='top')

    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, save_filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to: {full_path}")

if __name__ == "__main__":
    # Ensure these paths point to your actual data directories
    path_to_phi_data = r'/home/brandon/git_repos/UCM_AD_ML/UCM_AD_ML/data/flux_only_bias_train_e3'
    path_to_delta_data = r'/home/brandon/git_repos/UCM_AD_ML/UCM_AD_ML/data/w1_only_bias_train_e3'
    full_savepath = r'/home/brandon/Documents/Quals/NTM/Presentation_Visualizations'

    generate_publication_figure_v2(
        phi_dir=path_to_phi_data,
        delta_dir=path_to_delta_data,
        save_path=full_savepath
    )